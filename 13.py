#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


A, B = sys.stdin.read().split('\n\n')

dots = tile(list(ints(A)), 2)

for s in lines(B):
    c = next(ints(s))
    is_x = 'x' in s

    ndots = set()

    for a, b in dots:
        if is_x: a = min(a, 2 * c - a)
        else: b = min(b, 2 * c - b)
        ndots.add((a, b))

    dots = ndots


mx, my = map(max, zip(*dots))

for y in range(my+1):
    print(''.join('😂' if (x, y) in dots else '  ' for x in range(mx+1)))
