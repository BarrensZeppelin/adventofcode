#!/usr/bin/env pypy3
# 95/51

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

galaxies = []

G = [l for l in lines()]
extra_row = [set(l) == {"."} for l in G]
extra_col = [set(G[y][x] for y in range(len(G))) == {"."} for x in range(len(G[0]))]
print(extra_col)

y = 0
x = 0
ex = 0
ey = 0
for l in G:
    x = ex = 0
    for c in l:
        if c == "#":
            p = Point.of(x+ex, y+ey)
            for o in galaxies:
                res += (p - o).manh_dist()
            galaxies.append(p)
        x += 1
        if x < len(extra_col) and extra_col[x]:
            ex += 1 * 1000000-1

    y += 1
    if y < len(extra_row) and extra_row[y]:
        ey += 1 * 1000000-1

prints(res)
