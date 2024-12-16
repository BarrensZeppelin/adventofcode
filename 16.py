#!/usr/bin/env pypy3
# 161/39
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = Grid(lines())

s = G.rev["S"][0]
e = G.rev["E"][0]

def adj(p):
    p, d = p

    np = p + DIR_NORTHNEG[d]
    if G(np) in ".E":
        yield (np, d), 1

    yield (p, (d + 1) % 4), 1000
    yield (p, (d - 1) % 4), 1000
    # for i in range(4):
    #     d = (d + 1) % 4
    #     yield

dists, prevs = dijkstra(adj, (s, 0))

res = min(d for p, d in dists.items() if p[0] == e)
md = res

_, Q, _ = bfs(prevs, *((e, i) for i in range(4)))

res = len({p for p, _ in Q})

submit(res)
