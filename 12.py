#!/usr/bin/env pypy3
# 164/27
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = Grid(lines())
adj = G.adj(pred=lambda x, y: G(x) == G(y))

vis = set()
for y, l in enumerate(G):
    for x, c in enumerate(l):
        p = Point.of(x, y)
        if p in vis:
            continue

        D, Q, _ = bfs(adj, p)
        perim = 0
        p2 = set()
        for p in Q:
            p2.update(((p + d), d) for d in DIR if (p + d) not in D)
            perim += sum((p + d) not in D for d in DIR)
        vis.update(Q)
        # res += perim * len(Q)

        sides = 0
        for pp in list(p2):
            if pp not in p2:
                continue

            p, d = pp
            sides += 1
            N = [p]
            for p in N:
                for nd in DIR:
                    np = p + nd
                    npp = (np, d)
                    if npp in p2:
                        p2.remove(npp)
                        N.append(np)

        res += sides * len(Q)



submit(res)
