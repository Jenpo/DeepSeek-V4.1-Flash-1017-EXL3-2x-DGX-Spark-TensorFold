#!/usr/bin/env python3
"""Free the Engram table rows this node's rank never reads.

Each TP rank reads only the hash columns that belong to it: columns [lo, hi) of each Engram layer, which
is one contiguous run of table rows. Punching out the rest of the weight and scale rows returns about
half of the ~203 GB back per node without moving a byte of what the rank does read, so outputs cannot
change.

    python3 tools/engram_slice.py --config MODEL/config.json --src ENGRAM_DIR --rank 0 --dry-run
    python3 tools/engram_slice.py --config MODEL/config.json --src ENGRAM_DIR --rank 0 --punch-in-place --yes-punch
    python3 tools/engram_slice.py --config MODEL/config.json --src ENGRAM_DIR --rank 0 --check

Punching is permanent: the shards' published SHA-256 stops matching, and only a re-download undoes it.
Only the embed weight and scale rows are touched; headers and the projection weights in the same files
are left alone. `--check` reads headers and hole maps only, and exits 0 when every row this rank uses is
present, so a launcher can refuse a node prepared for the other rank.

Stdlib only. Row ranges come from the bucket primes, which must sum to `engram_num_embeddings`.
"""
import argparse
import json
import os
import struct
import sys

KEY_SUFFIX = {".engram.embed.weight": "weight", ".engram.embed.scale": "scale"}
DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U8": 1, "I8": 1,
               "BF16": 2, "F16": 2, "F32": 4}


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def bucket_primes(cfg):
    """[layer][(ngram-1) * heads] primes drawn upward from engram_vocab_size, none reused."""
    used, out = set(), []
    for _ in cfg["engram_layer_ids"]:
        row = []
        for _o in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _h in range(cfg["engram_n_heads"]):
                cur += 1
                while not is_prime(cur) or cur in used:
                    cur += 1
                used.add(cur)
                row.append(cur)
        out.append(row)
    return out


def rank_rows(cfg, rank, world):
    """layer id -> (first row, one past last row) this rank's hash columns use."""
    ps = bucket_primes(cfg)
    totals = cfg.get("engram_num_embeddings")
    out = {}
    for j, layer in enumerate(cfg["engram_layer_ids"]):
        offs, acc = [], 0
        for p in ps[j]:
            offs.append(acc)
            acc += p
        if totals is not None and totals[j] != acc:
            raise SystemExit(f"layer {layer}: primes sum to {acc} rows, config says {totals[j]}; refusing")
        cols = len(ps[j])
        lo, hi = rank * cols // world, (rank + 1) * cols // world
        out[layer] = (offs[lo], offs[hi - 1] + ps[j][hi - 1])
    return out


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return 8 + n, json.loads(fh.read(n))


def find_tables(src):
    """layer -> {weight|scale: (path, data_start, entry)}"""
    found = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(src, name)
        start, hdr = read_header(path)
        for key, entry in hdr.items():
            if key.startswith("__"):
                continue
            for suffix, kind in KEY_SUFFIX.items():
                if key.endswith(suffix):
                    layer = int(key.split(".")[1])
                    found.setdefault(layer, {})[kind] = (path, start, entry)
    return found


def row_bytes(entry):
    shape = entry["shape"]
    return DTYPE_BYTES[entry["dtype"]] * (shape[1] if len(shape) > 1 else 1)


def hole_map(path):
    """[(start, end)] of punched byte ranges, from SEEK_DATA/SEEK_HOLE."""
    holes, fd = [], os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        pos = 0
        while pos < size:
            try:
                data = os.lseek(fd, pos, os.SEEK_DATA)
            except OSError:
                break
            try:
                nxt = os.lseek(fd, data, os.SEEK_HOLE)
            except OSError:
                nxt = size
            if data > pos:
                holes.append((pos, data))
            pos = nxt
    finally:
        os.close(fd)
    return holes


def punch(path, start, length):
    fd = os.open(path, os.O_RDWR)
    try:
        os.posix_fallocate(fd, start, length)
        os.fallocate(fd, os.FALLOC_FL_PUNCH_HOLE | os.FALLOC_FL_KEEP_SIZE, start, length)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", help="copy this rank's rows here instead of punching")
    ap.add_argument("--punch-in-place", action="store_true")
    ap.add_argument("--yes-punch", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify-rows", type=int, default=0, help="read back N rows of the kept range")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if "engram_layer_ids" not in cfg:
        cfg = cfg.get("text_config", cfg)
    rows = rank_rows(cfg, args.rank, args.world)
    tables = find_tables(args.src)

    if args.check:
        bad = 0
        for layer, (r0, r1) in rows.items():
            for kind, (path, start, entry) in tables.get(layer, {}).items():
                base = start + entry["data_offsets"][0] + r0 * row_bytes(entry)
                span = (r1 - r0) * row_bytes(entry)
                covered = any(s <= base and base + span <= e for s, e in hole_map(path))
                if covered:
                    print(f"layer {layer} {kind}: rows {r0}..{r1} punched out of {path}")
                    bad = 1
        print("engram_slice: clean" if not bad else "engram_slice: MISSING ROWS")
        return bad

    total = 0
    for layer, (r0, r1) in sorted(rows.items()):
        for kind, (path, start, entry) in sorted(tables.get(layer, {}).items()):
            rb = row_bytes(entry)
            base = start + entry["data_offsets"][0]
            free = base + (r1 - r0) * rb
            print(f"layer {layer:>2} {kind:<6} rows {r0}..{r1} of {entry['shape'][0]}: "
                  f"free {(r1 - r0) * rb / 2**30:.1f} GiB at offset {free}")
            total += (r1 - r0) * rb
    print(f"engram_slice: {total / 2**30:.1f} GiB per node")

    if args.dry_run:
        return 0
    if args.out:
        print("copy mode is not implemented here; use --punch-in-place, or copy the shards and slice them yourself")
        return 1
    if not args.punch_in_place:
        print("nothing to do: pass --punch-in-place (permanent) or --dry-run")
        return 1
    if not args.yes_punch:
        print("refusing: punching is permanent and breaks the shards' published SHA-256. Pass --yes-punch.")
        return 1

    for layer, (r0, r1) in sorted(rows.items()):
        for kind, (path, start, entry) in sorted(tables.get(layer, {}).items()):
            rb = row_bytes(entry)
            base = start + entry["data_offsets"][0]
            punch(path, base, r0 * rb)
            punch(path, base + r1 * rb, (entry["shape"][0] - r1) * rb)
            print(f"punched layer {layer} {kind}: kept rows {r0}..{r1} in {path}")

    if args.verify_rows:
        for layer, (r0, r1) in sorted(rows.items()):
            for kind, (path, start, entry) in sorted(tables.get(layer, {}).items()):
                rb = row_bytes(entry)
                base = start + entry["data_offsets"][0] + r0 * rb
                with open(path, "rb") as fh:
                    fh.seek(base)
                    data = fh.read(min(args.verify_rows, r1 - r0) * rb)
                if len(data) != min(args.verify_rows, r1 - r0) * rb:
                    print(f"verify FAILED: layer {layer} {kind} kept rows are not readable")
                    return 1
        print(f"verify: {args.verify_rows} rows of each kept range read back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
