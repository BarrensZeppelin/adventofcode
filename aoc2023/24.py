#!/usr/bin/env pypy3
# 194/8
import z4 as z3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

P = Point
stones = []
for l in lines():
    p, v = l.split(" @ ")
    p, v = [P.of(*map(int, l.split(","))) for l in (p, v)]
    stones.append((p, v))
# print(stones)

res = 0
LOW = 200000000000000
HI = 400000000000000
# LOW = 7
# HI = 27

s = z3.Solver()
P = z3.Reals("x1, y1, z1")
V = z3.Reals("xv1, yv1, zv1")
times = z3.RealVector("t", len(stones))
for t, (p1, v1) in zip(times, stones):
    s.add(t >= 0)
    for i in range(3):
        s.add(P[i] + t * V[i] == p1[i] + t * v1[i])

r = s.check()
assert r == z3.sat
m = s.model()
print(sum(m[p].as_long() for p in P))
exit()

for i, (p1, v1) in enumerate(stones):
    print(f"{i/len(stones):.2%}")
    s = z3.Solver()
    t1, t2 = z3.Reals("t1, t2")
    s.add(t1 >= 0, t2 >= 0)
    for p2, v2 in stones[:i]:
        s.push()
        for j in range(2):
            x1 = p1[j] + t1 * v1[j]
            s.add(LOW <= x1, x1 <= HI)
            s.add(x1 == p2[j] + t2 * v2[j])

        r = s.check()
        res += r == z3.sat
        s.pop()
        # print(r)
        # print(p1, p2, r, s.sexpr())
            # res += 1




prints(res)
