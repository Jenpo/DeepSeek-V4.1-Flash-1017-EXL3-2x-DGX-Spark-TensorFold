# Tuning: what the pinned engine actually reads

This page is the map between the settings in `config/` and the code they reach in the pinned engine
(`bertholomus/TensorFold` @ `629c524a0ea57512a5342abb498d1b7c25353343`, the `deepseek-v41-zig` branch).
Everything here was read out of that tree (`zig/src/families/dsv41/`, `zig/src/core/`, `zig/src/server/`).
Where a number is a hard-coded constant rather than a setting, it is marked **constant**: exporting the
name from the comments does nothing.

The practical rule that comes out of this page: **change rows, not row counts.** The pinned kernel kit
(`abe07c7d`) ships row-invariant kernels built for 16 rows; widening `ROUND_ROWS` past 16 leaves those
kernels and drops into the generic grouped path, so every published recipe on this engine keeps
`ROUND_ROWS=16` and widens the *prefill* window instead.

## 1. Where a round's rows come from

`model.Options.round_rows` defaults to `round.default_rows` (16) and accepts 16..48
(`round.max_rows = 48`); `Round.init` rejects anything below 16 with `error.BadRoundRows`, so **there is no
"smaller rounds" setting** — a 4-row round cannot be asked for.

At every `round.forward` the engine sets `tri.decode_rows = e.round_rows` (a `pub var`, restored by `defer`
after the round). That one assignment is the switch between the two decode paths:

| rows | path |
| --- | --- |
| `n <= decode_rows` and not ring mode | `one_split` (the 16-row kernels) |
| `n <= ring_extra` (16) | `ring_mode` |
| more than `decode_rows` | `grouped` (accepted for `1..128` rows; `m == 0 or m > 128 or ls.len > 8` → `error.NotADecodeWindow`) |
| `m > 128` in the prefill path (`prompt.mmRows`) | `exl3_prefill.matmul` |

So `ROUND_ROWS` is a **two-sided** setting: 16 keeps the whole decode on the row-invariant kernels,
and 17..48 sends every round into `grouped`/`one_split` shapes the kit has no captured variant for
(`TF_DSV41_ROWS_CAP=64` on other recipes is exactly this, with a matching AOT build; ours is not).

### Row count and the draft allowance are coupled

The allocator scores a window as `exp_tokens * live / (round_ms[rows-1] + draft_ms)`
(`core/lanes/allocate.zig:63`, `extendCosts` linearly interpolates and extrapolates outside the table).
`draft.round_ms` holds 24 entries, 30.0 ms … 109.2 ms, and `draft_ms = 3.5`. Changing `ROUND_ROWS`
without a matching `ROUND_MS` table makes the interpolator answer a number for a window the engine never
measures, and the allocator then picks depths that do not pay. This is why every recorded community launch
command on this pin passes `--round-rows 16` and leaves `--round-ms` alone.

Draft depth follows from the same table:

```
depth(live, drafts, round_rows) = min( min({5,5,3,3}[min(live,4)-1], drafts),
                                       round_rows / max(live,1) -| 1 )
```

`draft.max_rows = 80`, `max_streams = 16`, `default_streams = 4`, `max_block = 8`, `max_absorb = 48`.

## 2. Memory a round costs (R-scaled)

Per-round device buffers scale linearly with R. Sizing from the field list:

| buffer | bytes | R=16 | R=24 | R=32 |
| --- | --- | ---: | ---: | ---: |
| `po_` | `R*hl*attn_splits*hd*4` | 16.8 MB | 25.2 MB | 33.6 MB |
| `u` | `R*groups*wo_a[0].n*2` | 4.0 MB | 6.0 MB | 8.0 MB |
| `h`, `h_alt` | each `R*hc*d*2` | 1.3 MB ea | 2.0 MB ea | 2.6 MB ea |
| `gm`, `ga` | `R*world*d*4` | 0.65 MB ea | 1.0 MB ea | 1.3 MB ea |
| `logits` | `R*world*hh*4` | 2.1 MB | 3.2 MB | 4.2 MB |
| `taps` | `R*max(dspark_taps,1)*d*2` | 1.0 MB | 1.5 MB | 2.0 MB |
| `rd.gl` | R-scaled | 0.8 MB | 1.2 MB | 1.6 MB |
| **total** | | **26.5 MiB** | **39.8 MiB** | **53.1 MiB** |

`po_` is 63 % of it. This is why `ROUND_ROWS` is not free even before the kernel question: at 32 rows a
round costs 2× the transient memory of a 16-row round, on a box whose floor is a few GiB.

Fixed (not R-scaled) allocations nearby: `glue` is `int64 [glue_n=13, max_rows]`, `inputs` is
`int64 [5, R]` uploaded once per round through `setGlue`; `Chunk.gl = 16*(d/256)*experts*4`,
`Chunk.logits = cap*experts*4`, `chunk_rows = 2048`.

## 3. Row thresholds that are constants, not settings

These are `pub var`/`const` in the pinned tree. The `TF_DS_*` names in older comments (`TF_DS_DECODE_ROWS`,
`TF_DS_ROWMM_RB`, `TF_DS_TOPK_FUSED_MAX`, `TF_DS_PREFILL_CHUNK`, `TF_DS_IDX_RB`, `TF_DS_MARKOV_*`,
`TF_DS_HC_RB`, `TF_DS_DRAFT_FUSED`, `TF_DS_2D_*`, `TF_DS_PARALLEL_DEPTH`, `TF_DS_BUCKET_MIN`,
`TF_DS_EAGER_ABSORB`, `TF_DS_ENGRAM_SPLIT`) exist **only in comments** — exporting them is a no-op.

| constant | value | where |
| --- | --- | --- |
| `tri.decode_rows` | 16 (overwritten per round) | `tri_index.zig`, `tri_norm.zig`, `tri_attn.zig` |
| `attn_splits` | 8 | `tri_attn.zig` |
| `rowmm2_rows` | 2 | `tri_norm.zig` |
| `rowmm_rb` | 16 | `tri_norm.zig` |
| `rowmm_parts_min` | 1 | `tri_norm.zig` |
| `ring_extra` | 16 | `prompt.zig` |
| `raw_rows` | 64 | `prompt.zig` |
| `exact_rows` | 64 | `exl3_experts.zig` |
| `tri_index.tile` | 64 | `tri_index.zig` |
| `index_topk` | 512 | `tri_index.zig` |
| `topk_fused_max` | 4096 | `tri_index.zig` |
| `pruned_rows` | 512 | `prompt.zig` |
| `bucket_min` | 1024 (doubles to ≥ n) | `round.zig` |
| `chunk_rows` | 2048 | `model.zig`, `layer_main.zig`, `native.prompt_rows` |
| `window_rows` | 6 (`error.WindowTooWide` above) | `lanes.zig` |
| `prefetch_budget` | 16 MiB, ≤16 ranges | `round.zig` |

Only four files read the environment at this pin: `native.zig`, `model.zig` (`TF_DS_REPLAY_FLOOR`),
`lanes_main.zig` (`TF_DS_CLOCK_MARKS`), `rdma.zig` (`TF_RDMA_GID_INDEX`, default 5).

## 4. Settings that do reach code

| setting | default | reach |
| --- | --- | --- |
| `TF_DS_ROUND_ROWS` / `--round-rows` | 16 | `Round.init`, `tri.decode_rows` |
| `TF_DS_REPLAY_FLOOR` | 1 (recipe) | global `prompt.replay_floor`: a last chunk of ≤16 rows reads its window from the replay cut, not the previous request's keys |
| `TF_DS_CLOCK_MARKS` | 0 | `lanes_main.zig` phase clocks |
| `TF_RDMA_GID_INDEX` | 5 | `rdma.servedGidIndex`; must match `NCCL_IB_GID_INDEX` |
| `TF_AOT_WEAKEST` | unset | `aot.zig` picks the most general fitting variant instead of refusing |
| `TF_DS_PREFILL_INTERLEAVE_MS` | 250 | patch `0005`: decode rounds for live streams between a long prompt's chunks |
| `TF_DS_IDX_SPLIT` | unset → from token 131,072 | patch `0001`: prefill indexer rows split across the ranks |
| `TF_DS_ENGRAM_LANES` | off | patch `0002`: second AIO ring (bit-identical, 1.5–2.6 % slower — the drive is the limit) |
| `TF_RDMA_IDLE_S`, `TF_RDMA_NAP_US` | 2, 500 | patch `0010`: the RDMA proxy naps instead of spinning a core |
| `TF_DS_POOL`, `TF_DS_WINDOW` | = `CONTEXT` | patch `0008`: pool larger than the window |
| `TF_DS_PLAIN_ROWS` | 16 | patch `0026`: mixed packs only |

Prefill chunking is **not** a setting on this pin: `chunk_rows = 2048` is hard-coded in three places.
`TF_DSV41_PREFILL_CHUNK` / `_ROWS=4096` belong to the Python engine and to newer Zig forks with a 4K AOT;
on this pin they are inert. Widening prefill is a rebuild, not an export.

## 5. Sampling: the `top_k` landmine

The DeepSeek-V4.1 sampler (`dsv41/sampling.zig`) is a keyed host sampler written to match the reference
Python exactly (libm `exp`/`log`, numpy sum order). Its behaviour:

- `temperature <= 0` → **greedy** (argmax, first of equals).
- `top_k == 0` → `error.NotPortedYet`: the nucleus path (a request that sets only `top_p`) was never ported.
  **`top_k` 0 or -1 is not an "off" switch.** The request fails its decode round, and on this pin every
  request running at that moment fails with it. Leave `top_k` unset, or send 1–56.
- `top_k + margin > max_candidates` (64) → `error.TooManyCandidates`. Above ~56 the candidate list no
  longer fits.
- `top_p` masks the suffix once the prefix mass is reached; ties break to the lower token id.
- The seed is keyed, not random: without an explicit `seed` it is
  `sha256(",".join(prompt_ids) + "|<seed_salt>")` first 8 bytes little-endian `& ((1<<63)-1)`
  (`lanes.sampling.seedFor`, salt from `TENSORFOLD_SEED_SALT`, default 0). So a sampled reply is
  reproducible for the same prompt — "identical twice" does not prove greedy.

Server-side defaults: `TEMPERATURE=1.0`, `TOP_P=0.95`, `TOP_K=20`, and `--top-k`/`--top-p`/`--min-p`/
`--temperature` override, or the model's `generation_config.json` when the flags are absent.
`integer_fields` includes `top_k`, and a negative integer field is normalised to `"0"` — which is exactly
the value that fails above.

Greedy is also not free on long agentic contexts: a client that omits `temperature` inherits the server
default, and a greedy default can loop inside `reasoning_content` until the budget is gone, returning an
empty `content` with `finish_reason: "length"`. Prefer a non-greedy default and let clients ask for greedy.

## 6. Absorb notes: what other recipes do that we do not

Each row was checked against our pinned tree; "absent" means the code is newer than `629c524`.

| lever | where it lives | our status |
| --- | --- | --- |
| `PREFILL_ADAPT` (+`_ADAPT_GIB=4.5`, `_MIN` 512, `_UP_GIB` 0.5) | Python engine `memory.Adapt`; newer Zig forks | absent on this pin |
| `FLOOR_PRICE` (503 after 30 s at the floor), `FAILFAST` (exit 70) | Python engine | absent on this pin; our `memguard.sh` is the equivalent |
| `ROWS_CAP=64` | newer Zig forks with matching AOT | absent; our kit has 16-row kernels only |
| `PF_4K` / `PF_WS_SHARE` / `PF_WS_FLOOR_GIB` | newer Zig forks | absent; needs a 4K AOT |
| `SESSION_DISK` (NVMe session tier) | Python engine | our spill v2 (`SPILL_GIB`) is the Zig equivalent, off by default |
| `GRAPH_FLOOR=hold`, `CALIB=zig`, `GRAPHS_MAX`, `GRAPH_DIP_ROUNDS` | newer forks / Python engine | absent |
| two RoCE rails (MTU 9000, aligned GID) | recipe, no engine change | **we run it**; −8.7 % cold 125K prefill on the reference pair |
| GPU clock latch check before benchmarking | recipe, no engine change | **adopted** — see `docs/GPU-CLOCK.md` |
| `TF_DS_SOLO_OVERHEAD` / `TF_DS_NO_COPIES` | Blackwellboy PRs #33/#34 | ours are `0032` (same patch) and `0033`; `0031` is the tool-framing fix |
| `TF_SEED_RANDOM` / `TF_TOP_K_MAX` | tonyd2wild patch 0004 | genuinely absent; not adopted — `top_k` is bounded by `max_candidates` anyway |

## 7. Reproducing a measurement

Three conditions decide whether a number on a unified-memory box means anything:

1. **≥5 minutes of quiet after an image build or a `docker save | load`.** The page-cache and copy tail
   runs for minutes; measuring through it reads 20 % low on this hardware.
2. **`/health.live.connections == 0` before and after the sample.** A client that keeps calling in
   contaminates the sample; discard it rather than average it.
3. **≥2 clean samples per configuration.** A single point is not a conclusion.

Also: both GPUs at the same clock ceiling (TP=2 meets twice a layer, so the slower GPU sets the pace),
and a clock-latch check first (`docs/GPU-CLOCK.md`).
