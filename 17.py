#!/usr/bin/env pypy3
# 5/3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

G = [list(map(int, l)) for l in lines()]

res = 0

def adj(n):
    p, s = n
    di = Point.of(1, 0)
    if s:
        di = di.perp()

    for dd in (-1, 1):
        cost = 0
        np = p
        for d in range(10):
            np = np + di * dd
            if 0 <= np.x < len(G[0]) and 0 <= np.y < len(G):
                cost += G[np.y][np.x]
                if d >= 3:
                    yield ((np, not s), cost)
            else:
                break

D, _ = dijkstra(adj, (Point.of(0, 0), False), (Point.of(0, 0), True))
W, H = len(G[0]), len(G)
p = Point.of(W-1, H-1)
prints(min(D[(p, s)] for s in (False, True)))
# prints(res)
