# GPU clock latch on GB10: check before you trust a benchmark

A GB10 can sit in a low-clock state that `nvidia-smi` does not report as a problem. The embedded
controller (EC) latches it below the OS, and neither a reboot nor a soft shutdown clears it.

## Symptom

`nvidia-smi` shows P0, persistence on, application clocks set, no active clock-event reasons, a clean
kernel log — and the GPU runs at 631–949 MHz idle and under load. Two of four Sparks in one fleet were
in this state while the other two ran at 2177–2561 MHz.

## Check

A 15 s fp16 4096×4096 matmul burn, with `nvidia-smi` sampled at 12 s:

```bash
python3 tools/clockcheck.py --burn
```

Rule of thumb for a healthy GB10: **80 W+ at ~2.2–2.4 GHz, 75–90 TFLOPS fp16**. A latched one sits near
700–950 MHz and under 20 W.

Reference points from the fleet that found this:

| node | before | after |
| --- | --- | --- |
| Reddie | 890 MHz / 5.3 W idle | 84.0 TFLOPS at 2294 MHz / 94.6 W |
| Asusi | 637 MHz / 4.6 W idle | 76.6 TFLOPS at 2333 MHz / 90.6 W |
| Spark4 | — | 89.2 TFLOPS at 2301 MHz / 93.4 W |
| Bluey | — | 88.8 TFLOPS at 2177 MHz / 92.9 W |

## Fix

Shut down, unplug the power adapter for 30–60 s (longer is fine), plug it back in, power on. A reboot is
not enough.

## Why it matters here

TP=2 meets twice per layer, so the slower GPU sets the pace for the pair: one Spark locked at 1800 MHz
against 2200 MHz on the other cost 3–6 % of prefill and 3 % of decode in the reference measurements. A
latched GPU is far worse than that, and it makes an A/B against a healthy fleet read as an engine
regression. Run the burn check before any number leaves the machine.
