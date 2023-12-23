#!/usr/bin/env pypy3
# 22/59

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

G = lines()
W, H = len(G[0]), len(G)

D = dict(zip(">^<v", DIR_NORTHNEG))

junctions = {(1, 0), (W-2, H-1)}
for y in range(H):
    for x in range(W):
        if G[y][x] == "#":
            continue
        cnt = 0
        for nx, ny in neighbours(x, y):
            if 0 <= ny < H and G[ny][nx] != "#":
                cnt += 1
        if cnt > 2:
            junctions.add((x, y))

# print(len(junctions))
# exit()
jun_index = dict(zip(junctions, range(len(junctions))))

def g(x, y, prev, dist):
    if (x, y) in junctions:
        return jun_index[(x, y)], dist

    for nx, ny in neighbours(x, y):
        if G[ny][nx] != "#" and (nx, ny) != prev:
            return g(nx, ny, (x, y), dist+1)
    assert False

edges = []
for i, p in enumerate(junctions):
    for nx, ny in neighbours(*p):
        if 0 <= ny < H and G[ny][nx] != "#":
            edges.append((i, *g(nx, ny, p, 1)))

adj = {i: [] for i in range(len(junctions))}
for i, j, w in edges:
    adj[i].append((j, w))

starti = jun_index[(1, 0)]
endi = jun_index[(W-2, H-1)]
INF = 1<<60

@cache
def f(i, bs):
    if i == endi:
        return 0

    best = -INF
    for j, w in adj[i]:
        if bs & (1<<j):
            continue
        best = max(best, w+f(j, bs|(1<<j)))
    return best
prints(f(starti, 1<<starti))
exit()


@cache
def f(x, y, prev, bs):
    if y == H-1:
        return 0
        # return (0, ())

    # print(x, y)

    # c = G[y][x]
    # if c in D:
    #     dx, dy = D[c]
    #     nx, ny = x+dx, y+dy
    #     if (nx, ny) == prev:
    #         return -(1<<60)
    #     return 1+f(x+dx, y+dy, (x, y))

    # best = (-INF, ())
    best = -INF
    for nx, ny in neighbours(x, y):
        if G[ny][nx] != "#" and (nx, ny) != prev:
            nbs = bs
            if (nx, ny) in junctions:
                i = jun_index[(nx, ny)]
                if bs & (1<<i):
                    continue
                nbs |= (1<<i)
            rec = f(nx, ny, (x, y), nbs)
            # best = max(best, (1+rec[0], ((nx, ny), rec[1])))
            best = max(best, 1+rec)
    return best

res = f(1, 0, (0, 0), 0)
# pp = res[1]
# pth = []
# while pp:
#     pth.append(pp[0])
#     pp = pp[1]

# print(pth, len(pth))
# print_coords({(x, y): ("O" if (x, y) in pth else G[y][x]) for x in range(W) for y in range(H)})

prints(res)
