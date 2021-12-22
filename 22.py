#! /usr/bin/env pypy3
# 3/>100
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))


L = 50
cub = []
D = {}
for l in lines():
    x1, x2, y1, y2, z1, z2 = ints(l)
    on = int(l.startswith("on"))

    me = (x1, x2, y1, y2, z1, z2)

    ncub = []
    for pcub in cub:
        avoid = any(me[i + 1] < pcub[i] or me[i] > pcub[i + 1] for i in range(0, 6, 2))

        if avoid:
            ncub.append(pcub)
        else:
            for split_at in range(0, 6, 2):
                def f(cur, i):
                    if i == 6:
                        yield cur
                    else:
                        (x, y), (a, b) = me[i : i + 2], pcub[i : i + 2]
                        if i < split_at:
                            yield from f(cur + (max(x, a), min(y, b)), i + 2)
                        elif i > split_at:
                            yield from f(cur + (a, b), i + 2)
                        else:
                            if a < x:
                                yield from f(cur + (a, x - 1), i + 2)
                            if y < b:
                                yield from f(cur + (y + 1, b), i + 2)

                ncub.extend(f((), 0))

    cub = ncub
    if on:
        cub.append(me)

    for x in range(max(x1, -L), min(x2, L) + 1):
        for y in range(max(y1, -L), min(y2, L) + 1):
            for z in range(max(z1, -L), min(z2, L) + 1):
                D[(x, y, z)] = on

print(sum(D.values()))


res = 0
for x1, x2, y1, y2, z1, z2 in cub:
    res += (x2 + 1 - x1) * (y2 + 1 - y1) * (z2 + 1 - z1)

prints(res)
