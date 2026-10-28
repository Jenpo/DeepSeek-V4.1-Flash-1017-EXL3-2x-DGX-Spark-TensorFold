# Credits

This lane stands on other people's work. Named here by role.

## The model

- **DeepSeek-AI** — DeepSeek-V4.1-Flash and DeepSeek-V4.1-Flash-1017, MIT. The Engram tables come from the
  original checkpoint at `1bae0e10`.
- **LibertAIDAI** — the EXL3 2.9 bpw pack this lane serves (`c178a01284fc`).
- **turboderp** — exllamav3, whose EXL3 format the pack and the kernels use.

## The engine

- **Ash Hart** — TensorFold. Apache-2.0 from 0.6.0, MIT before that.
- **bertholomus** — the TensorFold fork carrying the DeepSeek-V4.1 family (`deepseek-v41-zig` branch), the
  TP2 kernel kit, the recipe skeleton this repository follows, and most of the patches in `patches/`.

## Fixes we carry

- **MiaAI-Lab** — the two-node recipe the patch series is ported from, and the GLM-5.3-Flash recipe that
  several of those patches originally came from. Both are Apache-2.0. MiaAI-Lab's *EXL3* two-node kit
  (`DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks`) is AGPL-3.0 and is **not** used anywhere in this repository.
- **Robert Wojciechowski** — the port series (`0011`–`0022`).
- **Coinupbtc** — prefill interleave (`0005`, upstream PR #5), tool-call prose (`0027`), plain request
  text (`0028`).
- **Manrique** — DSML tag scope (`0030`, upstream `4803479e8`).
- **Blackwellboy** — tool-only framing (`0031`, `6568611ea`), solo head overhead (`0032`, `b6e948d98`),
  Engram AIO leak (`0033`, `12132870b`), plus the two-rail and copy-window measurements this repository's
  `docs/RAILS.md` and `docs/TUNING.md` cite.
- **jayleaton** — the classic-line recipe whose knob tables and memory-floor discipline shaped `docs/MEMORY.md`.
- **x00byte** — the `LibertAIDAI` 2.9 bpw dual-Spark recipe on the Python engine: the eight-slot
  1,048,576-token pool shape and the effort-75 tuning comparison that corroborate `docs/TUNING.md`'s
  pool and effort findings from an independent build of the same checkpoint.
- **sfxnz** — the EXL3 vLLM recipe's lever-by-lever A/B record. Its measured refutations of several
  widely-copied levers are independent evidence for `docs/TUNING.md` §3: knobs that pay off on one
  engine's kernel stack are inert or negative on another, so a knob name is not a portable finding.
- **tonyd2wild** — the GB10 clock-latch diagnosis reproduced in `docs/GPU-CLOCK.md`, and the fleet
  measurements that first showed this pack's ceiling.
- **lucataco**, **swbiggart**, **mrsergiopascal**, **willjriley**, **nmcbride**, **LeetJoe**,
  **den13501**, **Capicua25x** — upstream TensorFold PRs and issues that the pinned engine and the newer
  forks build on.

## Tools

- **xgrammar** — structured output, via `libtfgrammar`.
- **NVIDIA** — the PyTorch container image and the GB10 platform.

If you contributed something this lane runs and are not named here, that is an omission, not a claim.
