#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()


Ps = [Point.of(*map(int, p.split(","))) for p in lines()]

N = len(Ps)
uf = UF()


es = []
for i, a in enumerate(Ps):
    for j in range(i+1, N):
        b = Ps[j]
        es.append(((a - b).dist2(), i, j))

es.sort()

comps = N

for _, i, j in es:
    if uf.join(i, j):
        comps -= 1
        if comps == 1:
            submit(Ps[i].x * Ps[j].x)
            exit()


for _, i, j in es[:1000]:
    uf.join(i, j)


sets = [[] for _ in range(N)]
for i in range(N):
    sets[uf.find(i)].append(i)

ls = sorted([*map(len, sets)])[-3:]
from math import prod

res = prod(ls)

submit(res)
