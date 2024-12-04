#!/usr/bin/env pypy3
# 103/43
from __future__ import annotations

from util import *

replace_stdin()

"""
L = sys.stdin.read().split("\n\n")

tile, rotate, Point, sign
"""

res = 0

G = lines()
H = len(G)
W = len(G[0])

for y in range(H):
    for x in range(W):
        if G[y][x] != "A" or y == 0 or y == H - 1 or x == 0 or x == W - 1:
            continue

        if {G[y-1][x-1], G[y+1][x+1]} == {"M", "S"} and {G[y-1][x+1], G[y+1][x-1]} == set("SM"):
            res += 1


        # for dx, dy in OCTDIR:
        #     for i, c in enumerate("XMAS"):
        #         nx = x + dx * i
        #         ny = y + dy * i
        #         if nx < 0 or nx >= W or ny < 0 or ny >= H or G[ny][nx] != c:
        #             break
        #     else:
        #         res += 1

submit(res)
