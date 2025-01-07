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

print_coords(dots)
