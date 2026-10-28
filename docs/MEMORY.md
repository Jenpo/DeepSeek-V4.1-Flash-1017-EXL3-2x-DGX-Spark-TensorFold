# Memory: the floor is the feature

A Spark's GPU shares its ~121 GiB with the host. A rank that takes all of it does not fail cleanly: the
machine stops answering and has to be power-cycled. That happened once in the reference fleet with the
engine's reserve lowered to 10 GiB, so the reserve is not a knob to trade for weights.

## What the numbers look like

- Weights: ~98.9 GiB per rank (routed experts 91.3, DSpark 3.5, the rest under 1 GiB each).
- Engine reserve: `TENSORFOLD_MEMORY_RESERVE_GIB`, 12.2 GiB. **Do not lower it.**
- KV: compressed FP4 with block scales, about 0.5 GiB at 262,144 tokens and ~1.8 GiB at the full window.
  It is sized from `CONTEXT` at load and shared by the streams.
- Everything else — a prompt's prefill workspace, the prompt cache, the CUDA graphs, the host's own
  software — draws from whatever `MemAvailable` still shows.

A head that also runs a desktop and containers has ~6 GiB less. With one running, the worst-case dip
during four parallel cold long prefills lands under a 2.5 GiB floor.

## Two guards

**`MIN_AVAIL_GIB` (112).** `start.sh` refuses to start unless each Spark has this much available.
Available here means `MemAvailable` plus the pages parked on the kernel's per-CPU free lists beyond their
resting level, which is how the engine's own startup budget counts it — an image build or a
`docker save | load` leaves several GiB parked for a minute or two. `MIN_AVAIL_WAIT_S` (120) re-reads the
memory for up to that long before refusing.

If it refuses, look at `free -g` and `grep Slab /proc/meminfo` and retry in two minutes. Do **not** set
`MIN_AVAIL_GIB=0` to get past it: that is the freeze.

**`memguard.sh`.** Runs on both Sparks while the server is up, samples every 0.25 s, and kills the rank's
container (kill first, log after) on either of two levels:

- `MemAvailable` under `MEMGUARD_HARD_GIB` (0.75 here) in two consecutive samples → immediate kill;
- under `MEMGUARD_GIB` (1.5 here) for `MEMGUARD_HOLD_S` (10 s) without once coming back over it.

A long prompt's workspace takes a few GiB for a moment and gives them back; that dip is logged with its
length and lowest value (`... and recovered: no kill`) and the rank lives. Our variant also runs a
reclaim pass first (`sync; echo 1 > /proc/sys/vm/drop_caches`) so a recoverable page-cache dip is not
charged to the engine.

It is a brake, not a guarantee: a very fast allocation can outrun it. Log:
`~/.cache/dsv41-tensorfold/logs/memguard.log`.

**A kill is not a crash.** When memguard fires, the head looks like the service vanished. Check the log
before diagnosing an engine fault. In one incident `MemAvailable 2489 MiB < 2.5 GiB for 4 s` was the whole
story, and the real gap was that rank 1 kept 116 GiB of a dead rank 0's pool without noticing.

## Admission reserves prompt and reply together

`prompt + max_tokens + ROUND_ROWS + 2` positions (18 more at 16-row rounds), aligned to 2,048. A request
that names no budget gets `MAX_TOKENS`. A request whose prompt and budget together exceed the pool is
refused (`PromptTooLong`), not queued. So the equal split of a pool is not `POOL / PARALLEL` of context:
at `POOL=2097152` with a 131,072-token reply budget, four equal streams fit about 393,000 tokens of
context each.

## Raising the pool

`POOL` above `CONTEXT` (patch `0008`) keeps more long prompts resident and lets more long requests run at
once, at about 0.9 GiB a rank per 1,048,576 positions, in host memory. Measure the head's headroom first.
A pool that the ranks cannot both hold fails at boot rather than at the first long prompt.
