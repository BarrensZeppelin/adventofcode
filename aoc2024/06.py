#!/usr/bin/env pypy3
# 631/211
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = lines()
H = len(G)
W = len(G[0])
for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c not in ".#":
            p = Point.of(x, y)
            print(p, x, y, c)

G = [list(l) for l in G]

class Done(Exception):
    pass

def nxt(s):
    p, d = s
    np = p + DIR_NORTHNEG[d]
    if not 0 <= np.x < W or not 0 <= np.y < H:
        raise Done
    if G[np.y][np.x] == "#":
        d  = (d - 1) % 4
    else:
        p = np

    return p, d

for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c in "^#":
            continue

        G[y][x] = "#"
        fg = fungraph((p, 1), nxt)
        try:
            fg[1<<30]
            res += 1
        except Done:
            pass
        G[y][x] = "."


# d = 1
# vis = {p}
# while True:
#     print(p)
#     np = p + DIR_NORTHNEG[d]
#     if not 0 <= np.x < W or not 0 <= np.y < H:
#         break
#     if G[np.y][np.x] == "#":
#         d  = (d - 1) % 4
#     else:
#         p = np
#         vis.add(p)

submit(res)
