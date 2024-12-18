#!/usr/bin/env pypy3
# 246/298
from __future__ import annotations

from util import *

replace_stdin()

res = 0

W, H = 71, 71
ps = []
for l in lines():
    a, b = ints(l)
    ps.append(Point.of(a, b))

def blocked(i):
    G = Grid.empty(W, H, ".")
    for p in ps[:i]:
        G.set(p, "#")
    adj = G.adj(pred=lambda x, y: G(x) != "#" and G(y) != "#")
    dists, _, _ = bfs(lambda p: adj.get(p, []), Point.of(0, 0))
    return Point((W-1, H-1)) not in dists

res = binary_search(blocked, 0, len(ps)+1)-1
print(res, ps[res])
res = ",".join(map(str, ps[res]))
submit(res)
