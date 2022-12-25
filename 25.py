#! /usr/bin/env pypy3
# 118
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


res = 0
S = "=-012"

for l in lines():
    v = 0
    for p, c in enumerate(l[::-1]):
        i = S.index(c) - 2
        v += 5**p * i
    print(l, v)
    res += v

print(res)

import z3

s = z3.Solver()

vars = []
z3s = 0
for i in range(30):
    x = z3.Int(f"d_{i}")
    s.add(-2 <= x, x <= 2)
    z3s += 5**i * x
    vars.append(x)

s.add(z3s == res)

assert s.check() == z3.sat

print("".join(S[s.model()[v].as_long()+2] for v in vars)[::-1])
