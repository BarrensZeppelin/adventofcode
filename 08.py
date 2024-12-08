#!/usr/bin/env pypy3
# 61/31
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = lines()
D = defaultdict(list)
for y, l in enumerate(G):
    for x, c in enumerate(l):
        D[c].append(Point.of(x, y))

W = len(G[0])
H = len(G)

uniq = set()
for c, l in D.items():
    if c == '.':
        continue
    for i, p1 in enumerate(l):
        for p2 in l[i + 1:]:
            d = p2 - p1
            from math import gcd
            g = gcd(d.x, d.y)
            d.x //= g
            d.y //= g

            for i in (-1, 1):
                pp = p1
                while True:
                    if not 0 <= pp.x < W or not 0 <= pp.y < H:
                        break
                    uniq.add(pp)
                    pp += d * i

            # for x in (p1 - d, p2 + d):
            #     if 0 <= x.x < W and 0 <= x.y < H:
            #         uniq.add(x)

res = len(uniq)

submit(res)
