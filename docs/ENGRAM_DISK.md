# Engram tables: 203 GB of rows nobody fits in RAM

The two Engram layers (1 and 14) hold about 384 M rows of 264 bytes each — roughly 101 GB per layer,
203 GB per node in the original checkpoint. They never go into memory: the engine reads the rows a
prompt's tokens hash to, straight from disk, ahead of the layer that needs them.

## Why the tables must be local

A 4 KiB random read over NFS costs ~940 µs here; the same read off local NVMe costs ~90 µs. With the
tables on NFS, prefill fell from ~2,000 tok/s to ~690–890 tok/s. Put the two shards on each Spark's own
NVMe, both nodes, before the first start. The engine's own Engram check refuses a start if the head's
tables lack rank 0's rows.

## What can be released

Each rank only needs its own hash heads: 24 heads of the 48, twelve of them per rank, split by the
`engram_heads` field of the rank-cache. The rows for the other rank's heads are never read by this node.
On the reference fleet, punching those rows out in place returned **94.4 GiB per node** (95 % → 84 % full).

```bash
python3 tools/engram_slice.py --config MODEL_DIR/config.json --src ENGRAM_DIR --rank 0 --world 2 \
        --punch-in-place --yes-punch --verify-rows 256
```

- `--dry-run` prints the row ranges and the bytes without touching anything.
- `--check` verifies that the ranges present are the ones the head expects; the launcher runs it before
  every start.
- Row ranges come from the bucket primes and must sum to `engram_num_embeddings`; only the embedding
  weight and scale rows are released, never anything else in the shard.
- `--out DIR` copies instead of punching, and needs ~95 GB free; `--punch-in-place` needs nothing.

## The irreversible part

Punching is permanent. It breaks the checkpoint's own shard SHA-256 check, so `setup.sh`-style
verification fails afterwards — the shard is no longer the file the hub published. The engine does not
care (it checks row ranges, not file hashes), but anything that re-downloads or re-verifies will.
If the shards are the only copy, use `--out` to a scratch volume instead, or keep a copy.

`ENGRAM_CHECK=0` skips the launcher's pre-start row check. Do not use it to get past a genuine failure.
