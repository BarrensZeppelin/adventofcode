#!/usr/bin/env pypy3
# 219/229
from __future__ import annotations

import networkx as nx
from util import *

replace_stdin()

res = 0

e = [l.split("-") for l in lines()]
adj = make_adj(e, both=True)

V = set()
for a in adj:
    for b, c in combinations(adj[a], 2):
        if b in adj[c]:
            t = tuple(sorted([a, b, c]))
            if t not in V:
                V.add(t)
                if any(x[0] == "t" for x in t):
                    res += 1


G =  nx.Graph()
for a, b in e:
    G.add_edge(a, b)
    G.add_edge(b, a)

best = max(nx.find_cliques(G), key=len)

submit(",".join(sorted(best)))


# uf = UF()
# for a, b in e:
#     uf.join(a, b)
#
# for s in uf.sets():
#     print(s)
#     if len(s) == 3 and any(x[0] == "t" for x in s):
#         res += 1



# submit(res)
