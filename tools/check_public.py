#!/usr/bin/env python3
"""Refuse to publish anything that names a machine, a person or a credential.

    python3 tools/check_public.py            scan tracked and untracked files
    python3 tools/check_public.py --history  also scan every commit's tree and message
    python3 tools/check_public.py --fix      print what would be masked (no file is changed)

Findings exit 1. A clean tree prints "check-public: clean" and exits 0.

What it looks for:
  private IPv4 (10/8, 172.16/12, 192.168/16, 100.64/7), Tailscale addresses, hostnames, MAC addresses,
  absolute home paths, ssh public keys, and the usual token shapes (sk-, hf_, gh*_, xox*-, AKIA...).
  A file's own line can opt out with the allowlist file (one regex per line, optional TAB + glob).

Stdlib only. This is a scrubber, not an audit: it catches the mistakes that get made, not every mistake.
"""
import argparse
import os
import re
import subprocess
import sys

PRIVATE_V4 = re.compile(r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
                        r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
                        r"|\b192\.168\.\d{1,3}\.\d{1,3}\b"
                        r"|\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")
TAILSCALE = re.compile(r"\bfd7a:[0-9a-f:]+\b|\.ts\.net\b")
MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
HOME_PATH = re.compile(r"(?<![\w/])/(?:home|Users)/([A-Za-z0-9._-]+)")
SSH_PUB = re.compile(r"ssh-(?:rsa|ed25519|dss) AAAA[0-9A-Za-z+/=]{40,}")
TOKENS = [
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
HOSTNAME = re.compile(r"\b(?:spark|dgx|gb10)[-_]?\d{1,2}\b|\bspark-[a-z]+\b")

# Paths that legitimately contain /home/<name>/ as documentation.
HOME_OK = {"user", "you", "<you>", "<user>", "ubuntu", "<name>"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "vendor"}
SKIP_EXT = {".log", ".out", ".pem", ".key", ".safetensors", ".bin", ".f32", ".pyc"}
SKIP_NAMES = {"id_rsa", "id_ed25519", ".env"}


def allowlist(root):
    path = os.path.join(root, "tools", "check-public-allow.txt")
    pats = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    pats.append(re.compile(line))
    return pats


def files(root, history=False):
    if history:
        out = subprocess.run(["git", "-C", root, "rev-list", "--all"],
                             capture_output=True, text=True, check=True).stdout.split()
        for sha in out:
            yield f"commit {sha[:12]}", subprocess.run(
                ["git", "-C", root, "show", "--format=%B", "--no-color", sha],
                capture_output=True, text=True).stdout
        return
    tracked = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True)
    for name in tracked.stdout.split():
        yield name, None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if rel not in tracked.stdout.split():
                yield rel, None


def mask(s):
    return s[:4] + "<redacted>" if len(s) > 4 else "<redacted>"


def scan_text(text, pats):
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(p.search(line) for p in pats):
            continue
        for rx, label in [(PRIVATE_V4, "private-ipv4"), (TAILSCALE, "tailscale"), (MAC, "mac"),
                          (SSH_PUB, "ssh-pubkey"), (HOSTNAME, "hostname")] + \
                         [(t, "token") for t in TOKENS]:
            for m in rx.finditer(line):
                hits.append((lineno, label, mask(m.group(0))))
        for m in HOME_PATH.finditer(line):
            if m.group(1) not in HOME_OK:
                hits.append((lineno, "home-path", mask(m.group(0))))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    pats = allowlist(root)

    findings = 0
    for name, text in files(root, args.history):
        if text is None:
            if os.path.splitext(name)[1] in SKIP_EXT or os.path.basename(name) in SKIP_NAMES:
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
                continue
        for lineno, label, masked in scan_text(text, pats):
            findings += 1
            print(f"{name}:{lineno}: {label}: {masked}")
    if findings:
        print(f"\ncheck-public: {findings} FINDINGS above")
        return 1
    print("check-public: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
