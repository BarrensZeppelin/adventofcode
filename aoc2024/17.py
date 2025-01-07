#!/usr/bin/env pypy3
# 391/2169
from __future__ import annotations

import z4 as z3

from util import *

replace_stdin()

res = 0

A, B, C, *P = ints()

o = z3.Optimize()
I = z3.BitVecs("A0 B0 C0 I0 O0", 64)
IA = I[0]
R = dict(zip("ABCIO", I, strict=False))
o.add(I[1] == 0, I[2] == 0, I[3] == 0, I[4] == 0)
o.add(I[0] >= 0)

pvec = [z3.BitVecVal(p, 64) for p in P]


def matchp(O):
    c = z3.BitVecVal(-1, 64)
    for pi, p in enumerate(P):
        c = z3.If(z3.BitVecVal(pi, 64) == O, pvec[pi], c)
    return c


for _ in range(170):
    NR = dict(zip("ABCIO", z3.BitVecs(f"A{_ + 1} B{_ + 1} C{_ + 1} I{_ + 1} O{_ + 1}", 64), strict=True))

    c = R["O"] == len(P)
    for pi in range(0, len(P), 2):
        conds = {c: R[c] == NR[c] for c in "ABCIO"}
        p, op = P[pi : pi + 2]
        cop = op
        if op >= 4 and op != 7:
            cop = R["ABC"[op - 4]]

        conds["I"] = NR["I"] == pi + 2
        if p == 0:
            conds["A"] = NR["A"] == R["A"] >> cop
        elif p == 1:
            conds["B"] = NR["B"] == R["B"] ^ op
        elif p == 2:
            conds["B"] = NR["B"] == cop % 8
        elif p == 3:
            conds["I"] = z3.If(R["A"] != 0, NR["I"] == op, NR["I"] == pi + 2)
        elif p == 4:
            conds["B"] = NR["B"] == R["B"] ^ R["C"]
        elif p == 5:
            conds["O"] = z3.And(R["O"] < len(P), NR["O"] == R["O"] + 1, cop % 8 == matchp(R["O"]))
            # conds["O"] = matchp(NR["O"])
            # conds["O"] = NR["O"] == cop % 8
            # O.append(cop % 8)
        elif p == 6:
            conds["B"] = NR["B"] == R["A"] >> cop
        elif p == 7:
            conds["C"] = NR["C"] == R["A"] >> cop
        else:
            assert False

        c = z3.If(R["I"] == pi, z3.And(*conds.values()), c)

    o.add(c)
    R = NR

print("DONE")
c = o.minimize(IA)
print("CHECKING")
r = o.check()
assert r == z3.sat, r
m = o.model()
prints(m[IA].as_long())
# prints(c.value().as_long())

# O = []
# i = 0
# while i < len(P):
#     p = P[i]
#     op = P[i+1]
#     cop = op
#     if op >= 4 and op != 7:
#         cop = R["ABC"[op-4]]
#
#     if p == 0:
#         R["A"] //= 1 << cop
#     elif p == 1:
#         R["B"] ^= op
#     elif p == 2:
#         R["B"] = cop % 8
#     elif p == 3:
#         if R["A"]:
#             i = op
#             continue
#     elif p == 4:
#         R["B"] ^= R["C"]
#     elif p == 5:
#         O.append(cop % 8)
#     elif p == 6:
#         R["B"] = R["A"] // (1 << cop)
#     elif p == 7:
#         R["C"] = R["A"] // (1 << cop)
#     else:
#         assert False
#
#     i += 2

# print(run(117440), P)

# submit(",".join(map(str, O)))
