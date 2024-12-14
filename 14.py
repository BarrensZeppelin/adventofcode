#!/usr/bin/env pypy3
# 450/1325
from __future__ import annotations

import z4 as z3
from util import *

replace_stdin()


res = 0

W, H = 101, 103
# W, H = 11, 7
C = Counter()
R = []
for l in lines():
    px,  py, vx, vy = ints(l)
    p = Point([px, py])
    v = Point([vx, vy])
    R.append((p, v))

# def p_at(t: int):
#     D = [p + v*t for p, v in R]
#     for p in D:
#         p.x %= W
#         p.y %= H
#
#     print()
#     print(t)
#     print_coords([(p.x, p.y) for p in D], ".")


# s = z3.Solver()
# T = z3.Int("T")
# s.add(T >= 0, T < 10**12)
# for i, (p1, v1) in enumerate(R):
#     for j, (p2, v2) in enumerate(R):
#         if j >= i:
#             break
#
#         print(i, j)
#         for p3, v3 in R[:j]:
#             with s:
#                 s.add((p1.x + v1.x * T) % W + 1 == (p2.x + v2.x * T) % W)
#                 s.add((p1.y + v1.y * T) % H + 1 == (p2.y + v2.y * T) % H)
#                 s.add((p2.x + v2.x * T) % W + 1 == (p3.x + v3.x * T) % W)
#                 s.add((p2.y + v2.y * T) % H + 1 == (p3.y + v3.y * T) % H)
#
#                 if s.check() == z3.sat:
#                     t = s.model()[T].as_long()
#                     p_at(t)


for i in range(10**8):
    for j, (p, v) in enumerate(R):
        p += v
        p.x %= W
        p.y %= H
        R[j] = (p, v)


    V = {p for p, v in R}
    # adj = {p: [p2 for p2 in p.neigh(OCTDIR) if p2 in V] for p in V}
    # adj2 = {p: [p2 for p2 in V if (p - p2).manh_dist() < 5] for p in V}
    # comp = conn_components(adj2)
    if len(V) == len(R):
        print(i+1)
        print_coords([(p.x, p.y) for p, v in R], ".")
        break

exit()
C2 = Counter()
h, w  = H//2, W//2
for p, c in C.items():
    if p.x == w or p.y == h:
        continue
    C2[(p.x//(w+1), p.y//(h+1))] += c

print_coords({p: str(v) for p, v in C.items()}, ".")
print(C, C2)

res = 1
for c in C2.values():
    res *= c

submit(res)
