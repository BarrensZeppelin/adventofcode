#! /usr/bin/env pypy3
# 69/54
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

G = [l.rstrip() for l in sys.stdin]
H = len(G)
W = len(G[0])

D = {(x, y): c for y, l in enumerate(G) for x, c in enumerate(l) if c != '.'}

def printr(D):
    L = [['.'] * W for _ in range(H)]
    for (x, y), c in D.items():
        L[y][x] = c

    for l in L:
        print(''.join(l))
    print()

for step in range(10 ** 10):
    moved = False

    for C, (dx, dy) in zip('>v', ((1, 0), (0, 1))):
        ND = {}
        for (x, y), c in D.items():
            nx, ny = (x + dx) % W, (y + dy) % H
            if c == C and (nx, ny) not in D:
                moved = True
                ND[(nx, ny)] = c
            else:
                ND[(x, y)] = c

        D = ND
        #printr(D)

    if not moved:
        prints(step+1)
        break


