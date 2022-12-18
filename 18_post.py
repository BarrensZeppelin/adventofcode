#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

cubes = set(Point([*ints(l)]) for l in lines())

dirs = [
    d for t in product(range(-1, 2), repeat=3) if (d := Point(list(t))).manh_dist() == 1
]

mins = [min(p[i] for p in cubes) for i in range(3)]
maxs = [max(p[i] for p in cubes) for i in range(3)]

res = sum(sum(p + d not in cubes for d in dirs) for p in cubes)
print(f"Part 1: {res}")

res = 0


def dfs(p: Point[int], V=set()):
    global res

    if not all(mins[i] - 1 <= v <= maxs[i] + 1 for i, v in enumerate(p)) or p in V:
        return

    V.add(p)
    for d in dirs:
        np = p + d
        if np in cubes:
            res += 1
        else:
            dfs(np)


dfs(Point([v - 1 for v in mins]))
print(f"Part 2: {res}")
