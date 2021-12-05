#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('_post.py', '.in'))

O = Counter()

def f(s) -> Point[int]:
    return Point.of(*ints(s))

for p1, p2 in map(lambda s: map(f, s.split(' -> ')), lines()):
    d = Point.of(*map(sign, p2 - p1))

    #if not (p1.x == p2.x or p1.y == p2.y): continue

    p = p1
    while True:
        O[p] += 1
        if p == p2: break
        p += d

print(sum(v > 1 for v in O.values()))
