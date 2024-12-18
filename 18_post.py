#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

replace_stdin()

res = 0

W, H = 71, 71
P = Point
ps = [P(ints(l)) for l in lines()]
G = Grid.empty(W, H, ".")
first = set(ps[:1024])
sp, ep = P.of(0, 0), P.of(W - 1, H - 1)
D = bfs(G.adj(pred=lambda x, y: not first & {x, y}), sp)[0]
print(f"Part 1: {D[ep]}")

uf = UF[P[int]]()
sps = set(ps)
for p in chain(set(G.points()) - sps, reversed(ps)):
    sps.discard(p)
    for d in p.neigh():
        if G.inbounds(d) and d not in sps:
            uf.join(p, d)

    if uf.find(sp) == uf.find(ep):
        print(f"Part 2: {','.join(map(str, p))}")
        break
