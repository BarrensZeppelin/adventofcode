#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0

L = [1]

cyc = 0
y = 0
D = set()
for i, l in enumerate(lines()):
    if l == "noop":
        ex = [L[-1]]
    else:
        x = int(l.split()[1])
        ex = [L[-1], L[-1] + x]

    for x, s in enumerate(ex):
        j = cyc + x
        rs = L[-1]
        rx = j % 40
        if abs(rx - rs) <= 1:
            D.add((rx, y))
        if j % 40 == 39:
            y += 1
        if j % 40 == 19:
            res += (j+1) * rs
        L += s,

    cyc += len(ex)

print(res)
print_coords(D)
