#! /usr/bin/env pypy3
# 16/44
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))


G = [list(map(int, l)) for l in lines()]
Q = [(0, 0, 0)]
INF  = 10 ** 10
H = len(G)
W = len(G[0])

for _ in range(4):
    G.extend([] for _ in range(H))

for dy in range(5):
    sy = dy * H

    for dx in range(5):
        if dx == dy == 0: continue

        dist = dy + dx

        for y in range(H):
            G[sy + y].extend((G[y][x] + dist - 1) % 9 + 1 for x in range(W))


W *= 5
H *= 5


D = [[INF] * W for _ in range(H)]
D[0][0] = 0

while Q:
    d, x, y = heappop(Q)
    if d > D[y][x]: continue

    if x == W-1 and y == H-1:
        print(d)

    for dx, dy in DIR:
        nx, ny = x + dx, y + dy
        if not 0 <= nx < W or not 0 <= ny < H: continue
        nd = d + G[ny][nx]
        if nd < D[ny][nx]:
            D[ny][nx] = nd
            heappush(Q, (nd, nx, ny))

