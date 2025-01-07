#!/usr/bin/env pypy3
# 116/81
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = Grid(lines())
sp, ep = G.crev["S"][0], G.crev["E"][0]

adj = G.adj(pred=lambda x, y: "#" not in (G(x), G(y)))
SD = bfs(adj, sp)[0]
ED = bfs(adj, ep)[0]

fdist = SD[ep]

C = Counter()
nonw = [p for p in G.points() if G(p) != "#"]
for p in G.crev["."] + [sp, ep]:
    for p2 in nonw:
        d2 = (p - p2).manh_dist()
        if d2 > 20:
            continue
        nd = SD[p] + d2 + ED[p2]
        # print(p, nd, p2)
        saved = fdist - nd
        if saved > 0:
            C[saved] += 1

        if saved >= 100:
            res += 1

print(sorted(C.items()))

submit(res)
