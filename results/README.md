# Results

Raw output from `tools/bench.py` and `tools/needle.py` on the lane in `config/lane.env.example`, thinking
off, temperature 0, both GPUs pinned to the same clock ceiling. Hostnames are replaced with `<HEAD_HOST>`;
nothing else is edited.

| file | what |
| --- | --- |
| `bench-20261028-final.json` | prefill 32K/98K and decode, 4 patches, clean window |
| `bench-0141.json` | prefill 32K/98K and decode, earlier run |
| `multi-0141.json` | 1/2/4 concurrent streams, aggregate and per-stream |
| `ctx1M-longprefill.json` | 30K and 100K prefill |
| `needle-512K.json` | a passphrase at 500K depth, 3/3 |

These are single-run samples on one pair, not matched A/Bs against other recipes. A repeated run picks a
new prompt salt so it does not read the prefix cache; the salt is recorded in each file.
