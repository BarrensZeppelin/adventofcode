#! /usr/bin/env pypy3
# 35/54
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0

cubes = set(Point.of(*map(int, l.split(","))) for l in lines())

diffs = []
for dx, dy, dz in product(range(-1, 2), repeat=3):
    d = Point.of(dx, dy, dz)
    if d.manh_dist() != 1: continue
    diffs.append(d)

mins = [1000] * 3
maxs = [-1000] * 3

for a in cubes:
    for i, v in enumerate(a):
        mins[i] = min(mins[i], v)
        maxs[i] = max(maxs[i], v)
    for d in diffs:
        if a + d not in cubes:
            res += 1


print(res)
res = 0

def free(p, V):
    if p in cubes: return False
    for i, v in enumerate(p):
        if v < mins[i] or v > maxs[i]:
            return True

    for d in diffs:
        np = p + d
        if np not in V:
            V.add(np)
            if free(np, V):
                return True

    return False

for a in cubes:
    for d in diffs:
        if free(a + d, {a+d}):
            res += 1


prints(res)
