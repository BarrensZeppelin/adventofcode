#!/usr/bin/env pypy3
# 149/29
from __future__ import annotations

from util import *

replace_stdin()

res = 0

import z4 as z3

s = z3.Optimize()
L = sys.stdin.read().split("\n\n")
P = z3.IntVector("P", 2)
s.add(P[0] >= 0)
s.add(P[1] >= 0)
for sc in L:
    a, b, p = map(ints, lines(sc))
    assert len(a) == 2 == len(b) == len(p)
    p[0] += 10000000000000
    p[1] += 10000000000000

    s.push()
    s.add(P[0] * a[0] + P[1] * b[0] == p[0])
    s.add(P[0] * a[1] + P[1] * b[1] == p[1])
    s.minimize(P[0] * 3 + P[1])

    if s.check() == z3.sat:
        m = s.model()
        res += m.eval(P[0] * 3 + P[1]).as_long()

    s.pop()


submit(res)
