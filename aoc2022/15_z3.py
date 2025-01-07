#! /usr/bin/env python3
import z3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_z3.py', '.in'))

#Y = 10
Y = 2000000

MAX = 4000000
#MAX = 20
MAXM = 4000000
beacons = set()
bad = set()

solver = z3.Solver()
x, y = z3.Ints("X Y")
for v in (x, y):
    solver.add(0 <= v, v <= MAX)

def Abs(v):
    return z3.If(v < 0, -v, v)

for sx, sy, bx, by in map(ints, lines()):
    if by == Y: beacons.add(bx)

    dist = abs(by - sy) + abs(bx - sx)
    rem = dist - abs(sy - Y)
    bad.update(range(sx - rem, sx + rem + 1))

    solver.add(Abs(sx - x) + Abs(sy - y) > dist)

print(len(bad - beacons))

assert solver.check() == z3.sat
print(solver.model().eval(x * MAX + y))
