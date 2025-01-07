#!/usr/bin/env pypy3
# 91/74
from __future__ import annotations

from util import *

replace_stdin()

res = 0

keys = []
locks = []
L = sys.stdin.read().split("\n\n")
for l in L:
    l = lines(l)
    key = l[0].count("#") == 5
    if not key:
        l.reverse()

    hs = []
    for x in range(5):
        y = 0
        while y < len(l) and l[y][x] == "#":
            y += 1
        hs.append(y)

    (keys if key else locks).append(hs)


print(keys, locks)
for k in keys:
    for l in locks:
        if all(a + b <= 7 for a, b in zip(k, l)):
            res += 1

submit(res)
