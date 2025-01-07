#! /usr/bin/env pypy3
# 184/107
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

rock = set()

for coords in lines():
    ps = [Point(list(map(int, l.split(',')))) for l in coords.split(' -> ')]
    for i in range(1, len(ps)):
        b = ps[i]
        a = ps[i-1]
        d = b - a
        d2 = Point([sign(v) for v in d])

        rock.add(a)
        while a != b:
            a += d2
            rock.add(a)


MAXY = max(p.y + 2 for p in rock)
for x in range(-1000, 1000):
    rock.add(Point.of(x, MAXY))
print(MAXY)

sand = set()
t = 0
done = False
while True:
    p = Point.of(500, 0)
    if p in sand:
        prints(t)
        break
    while True:
        for dx, dy in ((0, 1), (-1, 1), (1, 1)):
            np = p + Point.of(dx, dy)
            if np in sand or np in rock: continue
            p = np
            if p.y > MAXY:
                prints(t)
                exit()
            break
        else:
            t += 1
            sand.add(p)
            break


