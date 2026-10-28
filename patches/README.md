# Engine patches

Applied in filename order, against `bertholomus/TensorFold` @
`629c524a0ea57512a5342abb498d1b7c25353343` (branch `deepseek-v41-zig`). `zig/Dockerfile` in the
upstream recipe applies the same sequence; here the list is flattened and documented.

Verify before building:

```bash
cd TensorFold && git checkout 629c524a0ea57512a5342abb498d1b7c25353343
for p in ../patches/00*.patch; do git apply --check "$p" || echo "FAILED: $p"; git apply "$p"; done
```

All 30 apply cleanly on that commit (checked 2026-10-28). They do **not** apply to newer engine commits:
the recipe carries its own cache-tier, spill and KV-sharding code, and upstream reimplemented the same
ideas under different names, so a repin is a port, not a rebase.

## Where these come from

- `0001`, `0002`, `0004`–`0010`, `0013`–`0026`: the MiaAI-Lab recipe (`DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`),
  itself ported from the GLM-5.3-Flash TensorFold recipe where noted in each file's header.
- `0011`, `0012`, `0023`: base-engine and startup-budget fixes.
- `0027`, `0028`, `0029`: Coinupbtc and Maciej Kowalik, via the upstream recipe.
- `0030`–`0033`: upstream TensorFold commits from Manrique and Blackwellboy, taken as patches because our
  pin predates them (`4803479e8`, `6568611ea`, `b6e948d98`, `12132870b`).
- `0003` (C1 cache placement) was dropped when the pin moved to `629c524`; it is not shipped here.

## The list

| patch | what it does | default |
| --- | --- | --- |
| `0001-c7-indexer-split` | splits the indexer's prompt key rows between the two ranks during prefill | on from token 131,072 (`TF_DS_IDX_SPLIT`) |
| `0002-c3-engram-lanes` | optional second AIO ring for a prompt chunk's Engram reads | off (`TF_DS_ENGRAM_LANES=2`) |
| `0004-call-gate` | enforces `tool_choice: required` / a named function in the lanes | on |
| `0005-prefill-interleave` | decode rounds for live streams between a long prompt's chunks (upstream PR #5, issue #3) | 250 ms (`TF_DS_PREFILL_INTERLEAVE_MS`) |
| `0006-image-prompt-cache` | a prompt with pictures resumes and keeps its state, keyed by picture content | on (`TF_DS_IMAGE_CACHE=0` disables) |
| `0008-pool-window` | a pool larger than the window | `POOL` > `CONTEXT` |
| `0010-rdma-idle-nap` | the RDMA proxy naps instead of spinning a core | 2 s / 500 µs (`TF_RDMA_IDLE_S`, `TF_RDMA_NAP_US`) |
| `0011-cuda-budget-parked-pages` | the startup budget counts pages parked on the kernel's per-CPU free lists (kernel 6.7+) | on |
| `0012-dsv41-base-fixes` | drops a stale Engram read-ahead after a cancelled pass; two further pinned-engine fixes | on |
| `0013-core-spill-store` | spill v2's disk block store, 2,048-token blocks shared by later turns and forks | on store, off use |
| `0014-core-cache-tier` | host side of spill v2 and continued chats | — |
| `0015-dsv41-kv-shard-attn` | KV sharding: each conversation's compressed latents stored once, on the rank that owns them | off (`KV_SHARD`) |
| `0016-dsv41-spill-blocks` | kept prompt states written to local disk and read back beside decoding streams | off (`SPILL_GIB`) |
| `0017-dsv41-kit-row-split` | 32-row rounds on the 16-row kit: a gate the kit cannot run as one launch becomes `_rowmm_parts` launches of ≤16 rows | off |
| `0018-dsv41-kv-shard-lanes` | the lanes side of KV sharding | off |
| `0019-dsv41-spill-kv-shard` | spill and KV sharding on one pair; `rank_wire` version 2 | off |
| `0020-dsv41-pair-broken` | a broken rank pair ends the server (exit 75) instead of hanging forever | on; `TF_DS_PAIR_BROKEN=degrade` opts out |
| `0021-dsv41-spill-pictures` | spill for prompts with pictures (manifest version 3) | off |
| `0022-dsv41-kv-shard-ring` | KV-sharded decode exchanges on the family's RDMA ring instead of NCCL collectives | off |
| `0023-server-prepare-memory` | a long prompt's preparation no longer runs rank 0 out of memory (base-engine fix) | on |
| `0024-mixed-pack` | mixed-precision pack support (`plain_tmp`, bf16 plain rows) | on |
| `0025-widen-bf16` | widens the checkpoint reader's BF16 path | on |
| `0026-plain-rows` | plain rows for mixed packs | 16 (`TF_DS_PLAIN_ROWS`) |
| `0027-tool-call-prose` | a reply that precedes a tool call returns that prose as `content` instead of dropping it | on |
| `0028-plain-request-text` | plain request text handling | on |
| `0029-anthropic-budget-default` | default thinking budget on the Anthropic-shaped route | on |
| `0030-dsml-tag-scope` | DSML blocks are parsed first; an invoke for an unoffered tool stays text; the streamer sees only the prose before the block | on |
| `0031-tool-only-framing` | a streamed tool-only reply no longer sends its framing whitespace as `content` | on |
| `0032-solo-head-overhead` | optional fixed overhead for a lone head window; the DeepSeek family sets it to its draft allowance | opt-in (`solo_head_overhead_ms`, default null) |
| `0033-dsv41-engram-aio-leak` | a refused Engram row id no longer leaks the AIO batch's buffers (and no longer panics on a negative id) | on |

The four `0030`–`0033` patches are what took tool-calling from 4/8 to 8/8 on the edge cases we test, and
they are byte-identical in effect to the same fixes carried by other recipes on newer pins. `0032` is the
one switch: it is off unless a family sets `solo_head_overhead_ms`.

Measured on this lane, the four together are performance-neutral within noise (59.2/58.5 vs 59.3 tok/s
decode across clean samples) — they are correctness, not speed.

## What we did not take

`TF_SEED_RANDOM` and `TF_TOP_K_MAX` (tonyd2wild patch 0004) exist on newer pins only. We do not need
them: `top_k` is bounded by the sampler's candidate list anyway, and the seed is already keyed from the
prompt (`docs/TUNING.md` §5).
