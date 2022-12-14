#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))

rock = set()

for coords in lines():
    ps = [Point([*map(int, l.split(','))]) for l in coords.split(' -> ')]
    for a, b in zip(ps, ps[1:]):
        d = b - a
        dist = d.manh_dist()
        d //= dist
        rock |= {a + d * i for i in range(dist+1)}


MAXY = max(p.y for p in rock) + 2

part1 = False
for t in range(1<<30):
    p = Point.of(500, 0)
    if p in rock:
        prints(t)
        break

    while p.y < MAXY-1:
        for dx in (0, -1, 1):
            np = p + Point.of(dx, 1)
            if np in rock: continue
            p = np
            break
        else:
            break

    rock.add(p)
    if p.y == MAXY-1 and not part1:
        part1 = True
        print(t)


