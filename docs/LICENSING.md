# Licensing

| what | licence |
| --- | --- |
| this repository's scripts, documents, patch files, tools | Apache License 2.0 (`LICENSE`) |
| TensorFold (the engine these patches modify) | Apache License 2.0 from 0.6.0; earlier code under MIT |
| the patch files in `patches/` | modifications of TensorFold, Apache-2.0 §4(b) — each file states the files it changes and why |
| the kernel kit (`abe07c7d`) | built from TensorFold and xgrammar; see the kit's own `KIT-LICENSES.md` |
| model weights (EXL3 pack) | MIT, per the DeepSeek-V4.1-Flash model release — **not distributed here** |
| Engram tables (original checkpoint shards) | MIT, per the DeepSeek-V4.1-Flash model release — **not distributed here** |
| base container image `nvcr.io/nvidia/pytorch:26.07-py3` | NVIDIA Deep Learning Container License |

## Notes

- Nothing here redistributes weights, the kit's binaries, or engine source. `deploy/` points at a
  checkpoint you download yourself; `patches/` are diffs against a commit you clone yourself.
- TensorFold's LICENSE and NOTICE travel with the engine checkout, not with this repository.
- Several patches originate from the MiaAI-Lab recipe, which is Apache-2.0; each patch keeps its
  original author and date in the file header. `CREDITS.md` lists everyone by name.
- Some recipes in this space carry AGPL-3.0 components (abliteration tooling, for instance). This lane
  runs none of them: `ABLIT=0` in `config/lane.env.example`.
