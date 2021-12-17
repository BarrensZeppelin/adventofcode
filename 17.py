#! /usr/bin/env pypy3
# 17/7
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

x1, x2, y1, y2 = ints(input())

best = -1000
r = 0
for yvel in range(y1, 300):
    for xvel in range(x2+1):
        vel = Point.of(xvel, yvel)
        p = Point.of(0, 0)

        hy = 0

        while True:
            p += vel
            vel.x -= sign(vel.x)
            vel.y -= 1

            hy = max(hy, p.y)

            if x1 <= p.x <= x2 and y1 <= p.y <= y2:
                best = max(best, hy)
                r += 1
                break

            if p.y < y1 or p.x > x2: break

print(best)
prints(r)
