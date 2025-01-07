#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('_post.py', '.in'))

O = Counter()

def f(s) -> Point[int]:
    return Point.of(*ints(s))

for p1, p2 in map(lambda s: map(f, s.split(' -> ')), lines()):
    diff = p2 - p1
    d = Point.of(*map(sign, diff))
    #if not (p1.x == p2.x or p1.y == p2.y): continue
    O.update(p1 + d * i for i in range(max(abs(diff))+1))

print(sum(v > 1 for v in O.values()))
