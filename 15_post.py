#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


G = [list(map(int, l)) for l in lines()]
Q = [(0, 0, 0)]
INF = 10 ** 10
H = len(G)
W = len(G[0])

RW, RH = W * 5, H * 5

D = [[INF] * RW for _ in range(RH)]
D[0][0] = 0

V = set(product(range(RW), range(RH)))  # valid tiles

while Q:
    d, x, y = heappop(Q)

    if (x+1, y+1) in ((RW, RH), (W, H)):
        prints(d)

    for nx, ny in neighbours(x, y, V=V):
        # It's simple to implicitly extend the graph in this way
        dist = nx // W + ny // H
        nd = d + (G[ny % H][nx % W] + dist - 1) % 9 + 1
        if nd < D[ny][nx]:
            D[ny][nx] = nd
            heappush(Q, (nd, nx, ny))

