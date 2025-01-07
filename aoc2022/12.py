#! /usr/bin/env pypy3
# 30/19
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

G = [list(l) for l in lines()]
H = len(G)
W = len(G[0])

D = {}
Q = []
for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c == "S":
            start = (x, y)
            c = l[x] = "a"
        elif c == "E":
            end = (x, y)
            l[x] = "z"
        if c == "a":
            start = (x, y)
            D[start] = 0
            Q.append(start)

for x, y in Q:
    d = D[(x, y)]
    if (x, y) == end:
        prints(d)
        break

    h = G[y][x]
    for nx, ny in neighbours(x, y):
        if 0 <= nx < W and 0 <= ny < H:
            nd = d + 1
            if nd < D.get((nx, ny), 1<<30) and ord(G[ny][nx]) - ord(h) <= 1:
                D[(nx, ny)] = nd
                Q.append((nx, ny))
else:
    assert False
