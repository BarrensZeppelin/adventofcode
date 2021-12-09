#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

M = [list(map(int, l)) for l in lines()]
H = len(M)
W = len(M[0])

Q = []
r = 0
for y, l in enumerate(M):
    for x, d in enumerate(l):
        m = 10
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x +dx, y + dy
            if not 0 <= nx < W or not 0 <= ny < H: continue
            m = min(m, M[ny][nx])

        if d < m:
            Q.append((x, y))
            r += d + 1

print(r)

B = []
V = [[False] * W for _ in range(H)]
for sx, sy in Q:

    NQ = [(sx, sy)]
    V[sy][sx] = True

    for x, y in NQ:
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x +dx, y + dy
            if not 0 <= nx < W or not 0 <= ny < H: continue
            m = M[ny][nx]
            if m == 9 or V[ny][nx]: continue
            V[ny][nx] = True
            NQ.append((nx, ny))

    B.append(len(NQ))

B.sort(reverse=True)
a, b, c = B[:3]
prints(a * b *c)
