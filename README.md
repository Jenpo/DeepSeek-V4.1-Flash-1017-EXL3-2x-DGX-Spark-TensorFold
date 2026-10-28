# DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Spark

A recipe for serving **DeepSeek-V4.1-Flash-1017** (EXL3 2.9 bpw) on two NVIDIA DGX Spark / GB10 nodes
with the **TensorFold** native engine (Zig, TP=2), over the direct ConnectX-7 link, one rank per Spark.

This repository is the configuration and tooling around a pinned engine. It ships no weights and no
engine source: `deploy/` starts a lane, `docs/` explains the numbers, `patches/` lists the engine changes
we carry, `tools/` measures and checks.

```
head (spark-001)  ── ConnectX-7 direct link (2 rails) ──  worker (spark-002)
 rank 0                                                    rank 1
 OpenAI API :8888                                          no listener
```

## What it serves

| | |
| --- | --- |
| model | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `c178a01284fc5c0e859d8668cffaac81ca8f4042` |
| Engram tables | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, shards 47 and 48 (~203 GB per node) |
| engine | `bertholomus/TensorFold` @ `629c524a0ea57512a5342abb498d1b7c25353343`, `deepseek-v41-zig` branch |
| kernel kit | `bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10` @ `abe07c7d`, 16-row AOT |
| served id | `DeepSeek-v4.1-Flash-EXL3` |
| window | 1,048,576 tokens (`CONTEXT`), pool 1,048,576 (`POOL`) |
| streams | 4 parallel requests (`PARALLEL`) |
| rows per round | 16 (`ROUND_ROWS`) |
| API | OpenAI-compatible on port 8888, plus `/health` and `/metrics` |

Thinking, image input, DSML tool calling, structured output (`response_format`, `guided_choice`,
`guided_regex`) and DSpark speculative decoding are on.

## Measured

All numbers below are from this recipe on a Spark pair with both GPUs pinned to the same 1800 MHz clock
ceiling, thinking off, temperature 0, and no other client attached. They are single-run samples, not
matched A/Bs against another recipe; treat them as what this lane does, not as a ranking.

**Prefill** (cold, one request, `tools/bench.py`):

| prompt | TTFT | rate |
| --- | ---: | ---: |
| 32,768 tok | 15.1 s | 2,081 tok/s |
| 98,304 tok | 71.0 s | 1,327 tok/s |
| 479,012 tok (needle, 500K) | 279.7 s | 1,712 tok/s |

**Decode**:

| cell | tok/s |
| --- | ---: |
| one stream, 256 tokens after a short prompt | 30.4 |
| one stream, sampled over a live window | 59–60 |
| 2 streams, aggregate | 88.5 |
| 4 streams, aggregate | 114.6 |

The single-stream cell is sensitive to what else is on the box: 30 tok/s and 59 tok/s are the same lane
at different moments, and the 4-stream aggregate is the number that matters for real use. Long context
costs decode: after a 128K prompt the rate falls to the 40s per stream.

**Needle in a haystack**: 3/3 at 500K (479,012 tokens actually sent).

**Quality**: `tools/exact.py` and `tools/toolcheck.py` pass — greedy continuations match the reference
Python engine token for token, and tool calls are correct on the DSML edge cases we test (offered tools
only, no leaked markup).

## Requirements

- Two DGX Spark / GB10 nodes, 128 GB each, ~121 GiB visible to the GPU.
- A direct ConnectX-7 link between them, both RoCE rails up (see `docs/RAILS.md`).
- Docker with the NVIDIA runtime; base image `nvcr.io/nvidia/pytorch:26.07-py3`.
- Passwordless ssh from the head to the worker.
- ~100 GB of weights, ~203 GB of Engram shards and ~95 GB of prepared data per node.
- Zig 0.17.0 and that image's CUDA toolchain to build the engine.
- `nvidia-smi -lgc 0,2200` on both nodes, and a clock-latch check (`docs/GPU-CLOCK.md`).

## Getting started

1. **Weights.** Download the EXL3 pack and the two Engram shards on both nodes. Nothing here
   redistributes them; see `docs/LICENSING.md`.
2. **Engine.** Clone `bertholomus/TensorFold`, check out the pinned commit, apply `patches/` in order,
   then build for the Spark:

   ```bash
   zig build test
   zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast \
       -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
   ```

3. **Kit.** Fetch the kernel kit at the pinned revision and verify it against its `MANIFEST`.
4. **Config.** `cp config/lane.env.example config/lane.env` and fill in the node addresses, paths and
   the RoCE device names for your pair. Every value still reading `CHANGE-ME-*` makes `deploy/lane`
   refuse to start, and `preflight` lists the ones you missed.
5. **Start.** `deploy/lane preflight` then `deploy/lane start`. First start writes the rank caches
   (~93 GB per node) and the token map; expect 20+ minutes for that on a cold cache.
6. **Check.** `deploy/lane check` sends one small completion and prints the model's answer.

## The parameters we actually run

`deploy/lane print` prints the two `docker run` lines below with your `config/lane.env` substituted.
This is the production lane on spark-001/spark-002 as of 2026-10-28 — not a suggested starting point,
the values the measured numbers in this README were taken with. Anything not listed keeps its
engine default.

**rank 0 (head, `spark-001`)** — serves the API:

```
tensorfold-native serve /model --backend cuda --host 0.0.0.0 --port 8888 \
  --name DeepSeek-v4.1-Flash-EXL3 --context 524288 --parallel 4 --round-rows 16 \
  --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --thinking \
  --spill-dir /spill --spill-gib 64 --spill-min-free-gib 50 --spill-min-tokens 8192
```

**rank 1 (worker, `spark-002`)** — holds the second half of the weights, no listener:

```
tf-dsv41-lanes /model /rank-cache 1 2 <HEAD_ADDR> 29620 /kit \
  --engram /engram --token-map /token-map/dsv41_token_map.json \
  --pool 1048576 --streams 4 --drafts 1 --graphs 1 --side 1 --prefetch 1 --aio 1 \
  --rdma rocep1s0f0,roceP2p1s0f0
```

**environment**, both ranks (`-e` on the container; the script sets these from `config/lane.env`):

| variable | value | why |
| --- | --- | --- |
| `TF_TP_WORLD` | 2 | two ranks, TP=2 |
| `TF_TP_MASTER` / `TF_TP_PORT` | head link address / 29620 | rank 1 rendezvouses here |
| `TF_RDMA_DEVICES` | `rocep1s0f0,roceP2p1s0f0` | both ConnectX-7 rails; one rail costs 8.7 % on a 125K cold prefill (`docs/RAILS.md`) |
| `TF_RDMA_GID_INDEX` | 3 | must match `NCCL_IB_GID_INDEX`; verify with `show_gids` |
| `TF_RDMA_IDLE_S` / `TF_RDMA_NAP_US` | 2 / 500 | how long a quiet rail sleeps, and for how long per nap |
| `TF_DS_POOL` | 1048576 | shared KV pool, tokens across all slots |
| `TF_DS_WINDOW` | 524288 | per-request ceiling; a head with nothing else on it can run 1048576 |
| `TF_DS_STREAMS` | 4 | equals `--parallel` |
| `TF_DS_REPLAY_FLOOR` | 1 | keeps the last round replayable across a prompt-cache hit |
| `TF_DS_GRAPHS` | 1 | CUDA graphs for the decode round |
| `TF_DS_HC_SIDE` | 1 | side-stream hyper-connections |
| `TF_DS_L2_PREFETCH` | 1 | L2 prefetch of the next layer's weights |
| `TF_DS_ENGRAM_AIO` | 1 | io_uring reads for Engram rows instead of the thread pool |
| `TF_DS_PREFILL_INTERLEAVE_MS` | 250 | decode slices interleaved into a long prefill |
| `TENSORFOLD_CUDA_KERNELS` | `/kit/aot` | the 16-row AOT kit |
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | the link interface |
| `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` | same as the RDMA pair above | NCCL must pick the same rails and GID |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 12.2 (engine default) | **do not lower** (`docs/MEMORY.md`) |

**why these and not the neighbours**: `--round-rows 16` because the pinned kit has 16-row kernels
(`docs/TUNING.md` §1–2); `--top-k 20` because 0 or -1 is a request-killer on this pin (§5);
`--context 524288` with `POOL 1048576` because the pool outlives any single request;
`--spill-gib 64` with `--spill-min-free-gib 50` so a spill can never start below the memory floor
(`docs/MEMORY.md`). Each of these has a section explaining the measurement behind it.

## Operating notes

- **`top_k` must be ≥ 1.** `top_k` 0 or -1 is not "off" on this engine: the sampler's nucleus path was
  never ported, so the request fails its decode round, and every request in flight at that moment fails
  with it. Leave `top_k` unset or send 1–56. See `docs/TUNING.md` §5.
- **Set `max_tokens`.** The server default is 32,768; a thinking reply that runs into the budget before
  finishing returns an empty `content` with `finish_reason: "length"`.
- **Keep one thinking setting per conversation.** Switching it mid-conversation invalidates the prompt
  cache and reprocesses the whole history (~a minute on a 150K-token conversation).
- **Keep `ROUND_ROWS=16`.** The pinned kit has 16-row kernels; widening rows moves every round onto the
  generic grouped path and needs a rebuilt AOT. Widen the prefill window instead, and see `docs/TUNING.md`.
- **Do not lower the memory guards.** `MIN_AVAIL_GIB=112` and `memguard.sh` exist because a Spark that
  takes all of its unified memory does not fail — it stops answering. See `docs/MEMORY.md`.
- **Engram tables must be on local NVMe**, both nodes. Over NFS prefill drops by more than half
  (`docs/ENGRAM_DISK.md`).
- **Secure the port.** Bind `HOST=0.0.0.0` only behind a firewall; set `API_KEY_FILE` when the port is
  reachable from other machines.

## Repository layout

```
deploy/     launcher (start, stop, status, check, preflight) and an env example
config/     the lane settings we run, as an example
patches/    engine changes we carry, in apply order, with a README naming each one
docs/       TUNING, MEMORY, RAILS, ENGRAM_DISK, GPU-CLOCK, LICENSING
tools/      bench, needle, exact, toolcheck, clock check, engram slice, publish scrubber
results/    the runs behind the numbers above
```

## Credits

The engine is **TensorFold** by Ash Hart, and the DeepSeek-V4.1 family support comes from
bertholomus's fork, which is also where the TP2 kernel kit, the recipe structure and most of the
patches here originate. Blackwellboy, jayleaton, tonyd2wild, lucataco, MiaAI-Lab and Robert
Wojciechowski each contributed fixes or measurements that this lane runs. `CREDITS.md` names them.

License: Apache-2.0 for the scripts, documents and patches here. TensorFold is Apache-2.0 from 0.6.0.
Weights are not distributed. Details in `docs/LICENSING.md`.
