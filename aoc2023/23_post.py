#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

INF = 1<<60
G = lines()
W, H = len(G[0]), len(G)

V = {(x, y) for x in range(W) for y in range(H) if G[y][x] != "#"}
D = dict(zip(">^<v", DIR_NORTHNEG))

@cache
def longest_path_1(cur, prev):
    if cur[1] == H-1: return 0

    c = G[cur[1]][cur[0]]
    return 1 + max((
        longest_path_1(nxt, cur)
        for nxt in neighbours(*cur, [D[c]] if c in D else DIR, V)
        if nxt != prev
    ), default=-INF)


print(f"Part 1: {longest_path_1((1, 0), (0, 0))}")

junctions = [(1, 0), (W-2, H-1)] + [
    p for p in V if len([*neighbours(*p, V=V)]) > 2
]
jun_index = {p: i for i, p in enumerate(junctions)}
N = len(junctions)

def find_junction(cur, prev):
    dist = 1
    while (j := jun_index.get(cur, -1)) == -1:
        dist += 1
        cur, prev = next(nxt for nxt in neighbours(*cur, V=V) if nxt != prev), cur
    return j, dist

adj = [[find_junction(np, p) for np in neighbours(*p, V=V)] for p in junctions]
(endi, endd), = adj[1]

def longest_path_2(i, bs):
    if i == endi: return endd
    return max((w+longest_path_2(j, bs|(1<<j)) for j, w in adj[i] if not bs&(1<<j)), default=-INF)

print(f"Part 2: {longest_path_2(0, 1)}")
