#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))

p1 = p2 = 0
for l1, r1, l2, r2 in map(ints, lines()):
    r1 *= -1; r2 *= -1
    p1 += (min(l1, l2), max(r1, r2)) in ((l1, r1), (l2, r2))
    p2 += max(l1, l2) <= min(r1, r2)

print(p1, p2, sep="\n")
