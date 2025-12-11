#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

res = 0

adj = defaultdict(list)

for l in lines():
    a, *bs = l.split()
    for b in bs:
        adj[a[:-1]].append(b)


Q, cyc = topsort(adj)
assert not cyc

S = "svr"
i = Q.index(S)
R = Counter()
R[(S, 0)] += 1
for j in range(i, len(Q)):
    v = Q[j]
    for bs in range(4):
        pv = R[(v, bs)]
        if v == "dac":
            bs |= 1
        if v == "fft":
            bs |= 2
        for y in adj[v]:
            R[(y, bs)] += pv

submit(R[("out", 3)])
