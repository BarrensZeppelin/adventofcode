#!/usr/bin/env pypy3
# 20

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

adj = defaultdict(set)

for l in lines():
    a, r = l.split(": ")
    for b in r.split():
        adj[a].add(b)
        adj[b].add(a)

import sys
from collections import defaultdict
sys.setrecursionlimit(1<<30)

def dinics(graph, s, t):
    assert s != t
    N = len(graph)
    cap = [defaultdict(int, adj) for adj in graph]

    def augment(i, of):
        if i == t: return of
        used, v = 0, valid[i]
        while v and used < of:
            j = v.pop(); c = cap[i][j]
            r = augment(j, min(of - used, c))
            used += r
            cap[i][j] -= r
            cap[j][i] += r
            if 0 < r < c: v.append(j)
        return used

    flow = 0
    while True:
        valid = [[] for _ in range(N)]
        level = [-1] * N
        level[s], Q = 0, [s]
        for i in Q:
            if i == t: break
            for j, c in cap[i].items():
                if level[j] == -1 and c:
                    level[j] = level[i] + 1
                    Q.append(j)
                if level[i] + 1 == level[j] and c:
                    valid[i].append(j)
        else: return flow, cap
        flow += augment(s, float('inf'))

all = set(adj)
for v in adj.values():
    all |= v

idx = {a: i for i, a in enumerate(all)}
G = [defaultdict(int) for _ in range(len(all))]
for a, v in adj.items():
    for b in v:
        G[idx[a]][idx[b]] = 1

for a in all:
    for b in all:
        if a == b: continue

        flow, cap = dinics(G, idx[a], idx[b])
        if flow == 3:
            Q = [idx[a]]
            seen = set(Q)
            for i in Q:
                for j, v in cap[i].items():
                    if v > 0 and j not in seen:
                        seen.add(j)
                        Q.append(j)

            prints(len(Q) * (len(all) - len(Q)))
            exit()
            break
        else:
            print("AAA")

prints(res)
