# Two RoCE rails on one ConnectX-7 port

One QSFP port of a Spark reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and
two RoCE devices per node:

```
enp1s0f0np0 / rocep1s0f0
enP2p1s0f0np0 / roceP2p1s0f0
```

Address both twins on each node (one subnet each is fine, the same MTU on both nodes) and the pair uses
them as two rails; one rail alone is one x4 link, about 112 Gb/s of the port's 200.

## Measured effect

On the reference pair, with everything else held fixed:

| cell | one rail | two rails |
| --- | ---: | ---: |
| cold prefill, 125K code prompt | 67.9 s | 62.0 s |
| cold prefill, 125K prose prompt | 67.2 s | 61.2 s |
| solo code decode, 125K context | 91.5 tok/s | 92.8 tok/s |
| solo prose decode, 125K context | 56.2 tok/s | 57.1 tok/s |

That is −8.7 % / −8.9 % cold prefill and +1.4–2.5 % decode at long context. Twin links measured
108.9 Gb/s each.

## The one requirement that bites

The RoCE GID index must agree across the twins, and with NCCL:

```
TF_RDMA_GID_INDEX=5
NCCL_IB_GID_INDEX=5
```

If the two disagree the ring opens on a GID that does not exist on the other side and the ranks stall
instead of failing. Check with `show_gids` on both nodes before the first start; a persistent
`nmcli` profile (`blackfold-rail2`, MTU 9000) keeps the addressing across reboots.

## A measurement trap

`/sys/class/net/*/statistics` does **not** count mlx5 RoCE traffic — it stays near zero even with both
rails saturated. The authoritative counters are per-port:

```bash
for d in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do
  printf '%s %s\n' "$d" "$(cat "$d")"
done
```

The values are in 4-byte units. On our pair both rails read within 0.1 % of each other, which is the
evidence that two rails are actually in use — not the netdev counters.
