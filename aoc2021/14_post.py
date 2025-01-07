#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


start, _, *rest = lines()
adj = dict(l.split(" -> ") for l in rest)

pairs = Counter(a + b for a, b in zip(start, start[1:]))

for steps in (10, 30):
    for i in range(steps):
        npairs = Counter()
        for p, cnt in pairs.items():
            npairs[p[0] + adj[p]] += cnt
            npairs[adj[p] + p[1]] += cnt

        pairs = npairs

    C = Counter(start[0])
    for p, cnt in pairs.items(): C[p[1]] += cnt
    print(max(C.values()) - min(C.values()))
