#! /usr/bin/env pypy3
# 48/48
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

S = sys.stdin.read().split('\n\n')

scan = []
for i, s in enumerate(S):
    L = lines(s)
    seen = []
    for l in L[1:]:
        seen.append(list(ints(l)))

    #scan.append((i, seen))
    scan.append(seen)
    print(seen)

"""
from collections import defaultdict

pts = defaultdict(set)
for p in scan[0][1]:
    pts[tuple(p)].add(0)
print(pts)
"""

here = [set(map(tuple, scan.pop(0)))]
#scan.pop(0)
P = [Point.of(0, 0, 0)]

def mods(scanner):
    for xyz in permutations(range(3)):
        for flip in range(2**3):
            nscan = []

            for p in scanner:
                np = [p[i] * (((flip >> j)&1) * 2 - 1) for j, i in enumerate(xyz)]

                nscan.append(np)

            yield nscan

def f(next):
    for me in mods(next):
        """
        for ref1 in pts:
            for ref2 in me:
                rref1 = Point.of(*ref1)
                rref2 = Point.of(*ref2)

                s2 = rref1 - rref2
                C = Counter()

                for i, ps in enumerate(me):
                    pp = s2 + Point.of(*ps)
                    C.update(pts.get(tuple(pp), []))

                if any(v >= 12 for v in C.values()):
                    return me, s2

        """
        for prev in here:
            for ref1 in prev:
                for ref2 in me:

                    rref1 = Point.of(*ref1)
                    rref2 = Point.of(*ref2)

                    s2 = rref1 - rref2

                    overlap = 0
                    for i, ps in enumerate(me):
                        if overlap + len(me) - i < 12: break

                        pp = s2 + Point.of(*ps)
                        if tuple(pp) in prev:
                            overlap += 1

                            if overlap >= 12:
                                return me, s2

import random

while scan:
    random.shuffle(scan)
    for i in range(len(scan)):
        print(i, len(scan))
        sc = scan[i]
        r = f(sc)
        if r is not None:
            scan.pop(i)
            me, s2 = r
            here.append({tuple(s2 + p) for p in me})
            P.append(s2)
            #for p in me:
            #    pts[tuple(s2 + p)].add(j)
            break
    else:
        assert False

S = set()
for ps in here: S |= ps

#print(len(pts))
print(len(S))


best = 0
for i in range(len(P)):
    for j in range(i):
        best = max(best, (P[i] - P[j]).manh_dist())
print(best)
