#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))

l = lines()

O = Counter()

def f(s):
    return map(int, s.split(','))

for s in l:
    a, b = s.split(' -> ')

    x1, y1 = f(a)
    x2, y2 = f(b)

    #if not (x1 == x2 or y1 == y2): continue

    if x1 == x2:
        dx = 0
    else:
        dx = 1 if x2 > x1 else -1

    if y1 == y2:
        dy = 0
    else:
        dy = 1 if y2 > y1 else -1

    x, y = x1, y1
    while True:
        O[(x, y)] += 1

        if (x, y) == (x2, y2): break

        x += dx; y += dy

print(sum(v > 1 for v in O.values()))
