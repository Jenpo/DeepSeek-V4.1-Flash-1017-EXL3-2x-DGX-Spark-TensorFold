#!/usr/bin/env python3
"""Is this GB10 running at its real clock? Run this before you trust any number.

    python3 tools/clockcheck.py            sample the clock and power now
    python3 tools/clockcheck.py --burn     a 15 s fp16 4096x4096 matmul, sampled at 12 s

A healthy GB10 draws 80 W+ at ~2.2-2.4 GHz and does 75-90 TFLOPS fp16. A latched one sits near
700-950 MHz under 20 W, and `nvidia-smi` reports nothing wrong. The fix is a power cycle with the
adapter out for 30-60 s (docs/GPU-CLOCK.md).

Needs torch (the container image has it). Stdlib only for the non-burn mode.
"""
import argparse
import subprocess
import sys


def smi(fields):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"nvidia-smi failed: {exc}")
        return None
    return [x.strip() for x in out.split(",")]


def sample():
    vals = smi(["clocks.sm", "power.draw", "temperature.gpu", "utilization.gpu"])
    if not vals:
        return None
    return {"sm_mhz": float(vals[0]), "power_w": float(vals[1]),
            "temp_c": float(vals[2]), "util_pct": float(vals[3])}


def burn():
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device")
        return None
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    for _ in range(3):          # warm up so the measurement is steady-state
        a @ b
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    n = 0
    while time.time() - t0 < 12.0:
        a @ b
        n += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tflops = 2 * 4096 ** 3 * n / elapsed / 1e12
    return {"tflops_fp16": tflops, "iters": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn", action="store_true")
    args = ap.parse_args()

    before = sample()
    if before:
        print(f"idle: {before['sm_mhz']:.0f} MHz, {before['power_w']:.1f} W, {before['temp_c']:.0f} C")

    if args.burn:
        res = burn()
        after = sample()
        if res and after:
            print(f"burn: {res['tflops_fp16']:.1f} TFLOPS fp16 at {after['sm_mhz']:.0f} MHz, "
                  f"{after['power_w']:.1f} W")
            healthy = after["power_w"] >= 60 and after["sm_mhz"] >= 1900 and res["tflops_fp16"] >= 60
            print("verdict:", "healthy" if healthy else "LATCHED or throttled - see docs/GPU-CLOCK.md")
            return 0 if healthy else 1
        return 1

    if before and before["power_w"] < 20 and before["util_pct"] < 5:
        print("idle figures are normal for an idle GPU; run --burn to check under load")
    return 0


if __name__ == "__main__":
    sys.exit(main())
