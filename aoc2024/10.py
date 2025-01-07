#!/usr/bin/env pypy3
# 129/87
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = Grid(lines())

adj = G.adj(pred=lambda a, b: int(G.at(b)) == int(G.at(a)) + 1)

for sp in G.rev["0"]:
    def f(p):
        if G.at(p) == "9":
            return 1
        return sum(f(q) for q in adj[p])

    res += f(sp)

    # d, _, _ = bfs(adj, sp)
    # res += sum(G.at(p) == "9" for p in d)


submit(res)
