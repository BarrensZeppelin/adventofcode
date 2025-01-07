#!/usr/bin/env python3
import z4 as z3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

stones = [(Point(vs[:3]), Point(vs[3:])) for l in lines() for vs in [[*ints(l)]]]

res = 0
LO = 200000000000000
HI = 400000000000000

if False:
    s = z3.Solver()
    t1, t2 = z3.Reals("t1 t2")
    s.add(t1 >= 0, t2 >= 0)
    for i, (p1, v1) in enumerate(stones):
        if i % 3 == 0: print(f"{i/len(stones):.0%}")
        xs = [p1[j] + t1 * v1[j] for j in range(2)]
        s.push()
        for v in xs: s.add(LO <= v, v <= HI)

        for p2, v2 in stones[:i]:
            s.push()
            for j, v in enumerate(xs):
                s.add(v == p2[j] + t2 * v2[j])

            res += s.check() == z3.sat
            s.pop()

        s.pop()
else:
    def lineInter(s1: Point, e1: Point, s2: Point, e2: Point):
        d = (e1 - s1).cross(e2 - s2)
        if d == 0: return (-(s1.cross2(e1, s2) == 0), Point.of(0, 0))
        p = s2.cross2(e1, e2)
        q = s2.cross2(e2, s1)
        return (1, (s1 * p + e1 * q) / d)

    for i, (p1, v1) in enumerate(stones):
        for p2, v2 in stones[:i]:
            r, p = lineInter(p1, p1 + v1, p2, p2 + v2)
            assert r != -1
            if r == 1:
                works = LO <= p.x <= HI and LO <= p.y <= HI and \
                        all(sign(v) == sign(d) for v, d in zip(v1.c[:2], p - p1)) and \
                        all(sign(v) == sign(d) for v, d in zip(v2.c[:2], p - p2))
                res += works


print(f"Part 1: {res}")

s = z3.Solver()
P = z3.RealVector("p", 3)
V = z3.RealVector("v", 3)
times = z3.RealVector("t", len(stones))
for t, (p, v) in zip(times, stones):
    s.add(t >= 0)
    for i in range(3):
        s.add(P[i] + t * V[i] == p[i] + t * v[i])

assert s.check() == z3.sat
print(f"Part 2: {s.model().eval(sum(P)).as_long()}")
