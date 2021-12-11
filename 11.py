#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

r = 0

G = [list(map(int, input())) for _ in range(10)]

for i in range(10 ** 10):
    V = [[False] * 10 for _ in range(10)]

    Q = []
    for y, l in enumerate(G):
        for x, c in enumerate(l):
            G[y][x] += 1
            if G[y][x] > 9:
                V[y][x] = True
                Q.append((x, y))

    for x, y in Q:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= 10 or ny >= 10: continue
                G[ny][nx] += 1

                if not V[ny][nx] and G[ny][nx] > 9:
                    V[ny][nx] = True
                    Q.append((nx, ny))

    r += len(Q)

    if len(Q) == 10 * 10:
        prints(i+1)
        exit()

    for x, y in Q:
        G[y][x] = 0

prints(r)
