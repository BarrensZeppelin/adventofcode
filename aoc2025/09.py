#!/usr/bin/env -S uv run
from __future__ import annotations

from bisect import bisect_left
from random import randint, shuffle

from util import *

replace_stdin()

"""
L = sys.stdin.read().split("\n\n")

Grid, tile, rotate, Point, sign
"""

res = 0

Ps = [Point.of(*ints(l)) for l in lines()]

hor = defaultdict(list)
ver = defaultdict(list)

assert area2(Ps) > 0

magic: set[Point[int]] = set()
N = len(Ps)
for i in range(N):
    a, b = Ps[i], Ps[(i + 1) % N]
    if a.y == b.y:
        hor[a.y].append(minmax((a.x, b.x)))
    else:
        assert a.x == b.x
        ver[a.x].append(minmax((a.y, b.y)))
        p0, p2 = Ps[(i - 1) % N], Ps[(i + 2) % N]
        assert a.x not in (p0.x, p2.x)
        if p0.x < a.x < p2.x or p0.x > a.x > p2.x:
            magic.add(a)
            magic.add(b)

hor = sorted(hor.items())
keys = [t[0] for t in hor]

sxs = sorted({x for p in Ps for x in (p.x - 1, p.x, p.x + 1)})
sys = sorted({x for p in Ps for x in (p.y - 1, p.y, p.y + 1)})


def inside(p: Point[int]):
    for y1, y2 in ver[p.x]:
        if y1 <= p.y <= y2:
            return True

    i = bisect_left(keys, p.y)
    if i < len(keys) and keys[i] == p.y:
        for x1, x2 in hor[i][1]:
            if x1 <= p.x <= x2:
                return True

    par = 0
    ms = 0
    for y, l in hor:
        if y >= p.y:
            break
        ms += Point.of(p.x, y) in magic
        for x1, x2 in l:
            par += x1 <= p.x <= x2
    assert ms % 2 == 0
    return (par - ms // 2) & 1 == 1


shuffle(Ps)
YS = (48393, 50355)

for i, a in enumerate(Ps):
    print(i, N)
    for b in Ps[i + 1 :]:
        d = abs(b - a)
        ar = (d.x + 1) * (d.y + 1)
        if ar <= res:
            continue

        x1, x2 = minmax((a.x, b.x))
        y1, y2 = minmax((a.y, b.y))

        i = bisect_left(sxs, x1)
        bad = False
        while i < len(sxs) and (x := sxs[i]) <= x2:
            if not inside(Point.of(x, y1)) or not inside(Point.of(x, y2)):
                bad = True
                break
            if y1 <= YS[0] + 1 <= y2 and not inside(Point.of(x, YS[0] + 1)):
                bad = True
                break
            if y1 <= YS[1] - 1 <= y2 and not inside(Point.of(x, YS[1] - 1)):
                bad = True
                break
            i += 1
        if bad:
            continue
        i = bisect_left(sys, y1)
        while i < len(sys) and (y := sys[i]) <= y2:
            if not inside(Point.of(x1, y)) or not inside(Point.of(x2, y)):
                bad = True
                break
            i += 1
        if bad:
            continue
        for _ in range(1000):
            x = randint(x1, x2)
            y = randint(y1, y2)
            if not inside(Point.of(x, y)):
                break
        else:
            print(a, b, ar)
            res = ar

    continue

    # corners = [Point.of(x, y) for x in (a.x, b.x) for y in (a.y, b.y)]

    # print(i, N)
    #
    # x2 = binary_search(lambda x: not inside(Point.of(x, a.y)), a.x) - 1
    # assert x2 >= a.x
    #
    # bad = False
    # if a.y >= 50355:
    #     dy = a.y - 50355
    #     dx = abs(a.x - x2)
    #     for y in keys:
    #         if not 50355 < y < a.y:
    #             continue
    #         for y in (y-1, y, y+1):
    #             if not inside(Point.of(a.x, y)) or not inside(Point.of(x2, y)):
    #                 bad = True
    #                 break
    #         if bad:
    #             break
    # elif a.y <= 48393:
    #     dy = 48393 - a.y
    #     dx = abs(a.x - x2)
    #     for y in keys:
    #         if not a.y < y < 48393:
    #             continue
    #         for y in (y-1, y, y+1):
    #             if not inside(Point.of(a.x, y)) or not inside(Point.of(x2, y)):
    #                 bad = True
    #                 break
    #         if bad:
    #             break
    #
    # if not bad:
    #     x = (dx+1) * (dy+1)
    #     print(a, b, x)
    #     res = max(res, x)
    # continue

    for b in Ps:
        if 94654 not in (a.x, b.x):
            continue

        d = b - a
        corners = [Point.of(x, y) for x in (a.x, b.x) for y in (a.y, b.y)]
        if (
            all(inside(p) for p in corners)
            and a != b
            and (all(p.y >= 50355 for p in corners) or all(p.y <= 48393 for p in corners))
        ):
            x = (d.x + 1) * (d.y + 1)
            print(a, b, x)
            res = max(res, x)

        # if a != b and a.y == b.y:
        #     if a.y >= 50355:
        #         dy = a.y - 50355
        #         dx = abs(a.x - b.x)
        #     elif a.y <= 48393:
        #         dy = 48393 - a.y
        #         dx = abs(a.x - b.x)
        #
        #     x = (dx+1) * (dy+1)
        #     print(a, b, x)
        #     res = max(res, x)


submit(res)
