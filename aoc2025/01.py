#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

replace_stdin()

"""
L = sys.stdin.read().split("\n\n")

Grid, tile, rotate, Point, sign
"""

res = 0
x = 50

for l in lines():
    d = -1 if l[0] == "L" else 1
    n = int(l[1:])
    for _ in range(n):
        x = (x + d) % 100
        res += x == 0

    # x = (x + d * n) % 100
    # print(l, x)
    # res += x == 0


submit(res)
