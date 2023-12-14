#!/usr/bin/env pypy3
# 311/26

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

G = [list(l) for l in lines()]

res = 0

seen = {}
def key(G):
    return "".join("".join(l) for l in G)

def k2(G):
    return tuple(tuple(l) for l in G)

dist = 0
A = [k2(G)]
while True:
    W = len(G[0])
    H = len(G)
    dist += 1
    for _ in range(4):
        for x in range(W):
            # rocks = sum(G[y][x] == "O" for y in range(len(G)))
            #
            for y in range(H):
                if G[y][x] == "O":
                    while y-1 >= 0 and G[y-1][x] == ".":
                        G[y][x], G[y-1][x] = G[y-1][x], G[y][x]
                        y -= 1
                    # res += y+1

        G = rotate(G, -1)

    k = key(G)
    if k in seen:
        break

    seen[k] = dist
    A.append(k2(G))
    # for y in range(len(G)-1, -1, -1):
        #     if G[y][x] != "#" and rocks:
        #         rocks -= 1
        #         res += y+1

ndist = dist - seen[k]
G = A[seen[k] + (1000000000 - seen[k]) % ndist]

H = len(G)
W = len(G[0])
for x in range(W):
    for y in range(H):
        if G[y][x] == "O":
            res += H-y

prints(res)
