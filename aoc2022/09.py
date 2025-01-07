#! /usr/bin/env pypy3
# 72/21
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0

h = Point.of(0, 0)

tails = [Point.of(0, 0) for _ in range(9)]
V = {tuple(tails[-1])}

for l in lines():
    d, i = l.split()
    i = int(i)
    d = "RULD".index(d)
    dx, dy = DIR[d]
    dp = Point.of(dx, dy)
    for _ in range(i):
        h += dp
        prev = h
        for j, t in enumerate(tails):
            diff = prev - t
            if h != t and max(abs(diff)) == 2:
                if abs(diff.x) == 2:
                    diff.x //= 2

                if abs(diff.y) == 2:
                    diff.y //= 2

                t += diff
            tails[j] = t
            prev = t

        V.add(tuple(t))
        #print(h, t)


prints(len(V))
