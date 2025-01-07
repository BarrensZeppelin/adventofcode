#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

x1, x2, y1, y2 = ints(input())

def tri(x: int) -> int:
    return x * (x+1) // 2

best = -1000
r = 0
for yvel in range(y1, abs(y1)):
    def ypos(t: int) -> int:
        return yvel * t - tri(t-1)

    start = binary_search(lambda t: ypos(t) <= y2, 1)
    for xvel in range(int(x1 ** .5), x2+1):
        def xpos(t: int) -> int:
            if t <= xvel:
                return xvel * t - tri(t-1)
            return tri(xvel)

        t = start
        while y1 <= ypos(t) <= y2:
            if x1 <= xpos(t) <= x2:
                r += 1
                if yvel >= 0:
                    best = max(best, tri(yvel))
                break

            t += 1

print(best)
prints(r)
