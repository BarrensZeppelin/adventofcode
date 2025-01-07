#! /usr/bin/env pypy3
# 123/140
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

shapes = """####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##""".split('\n\n')
shapes = [set(Point.of(x+2, y) for y, l in enumerate(lines(s)[::-1]) for x, c in enumerate(l) if c != ".") for s in shapes]


maxy = 0
res = 0

#shapes = cycle(shapes)
done = set()
for x in range(7):
    done.add(Point.of(x, maxy))

#dir = cycle(input())
dir = input()

shapi = 0
diri = 0

#D = {(shapi, diri, frozenset(x for x in range(7))): 0}
D = {}

MAX = 1000000000000
foundcyc = False
it = 0
extra = 0
while it < MAX:

    if not foundcyc:
        k = (shapi, diri, frozenset((p.x, p.y-maxy) for p in done if (maxy - p.y) <= 20))
        if k in D:
            pit, pmaxy = D[k]
            cyclen = it - pit

            print(it, pit, cyclen, maxy, pmaxy)

            rem = (MAX-it)
            remcyc = rem // cyclen

            it += remcyc * cyclen
            extra = (maxy - pmaxy) * remcyc
            foundcyc = True

        D[k] = (it, maxy)

#for it in range(1000000000000):
    #if it % 1000 == 0:
    #    print(it / MAX)
    shap = shapes[shapi]
    shapi += 1
    shapi %= len(shapes)
    rock = [Point.of(*l) for l in shap]
    mrocky = max(p.y for p in rock)
    #print(maxy)
    for p in rock:
        p.y += maxy+4

    #print(rock)

    while True:
        d = dir[diri]
        diri += 1
        diri %= len(dir)
        #d = next(dir)
        #print(d)
        dx = -1 if d == "<" else 1
        minx = 10
        maxx = -1

        ok = True
        for p in rock:
            minx = min(minx, p.x+dx)
            maxx = max(maxx, p.x+dx)
            ok &= (p + Point.of(dx, 0)) not in done

        if ok and 0 <= minx <= maxx < 7:
            for p in rock:
                p.x += dx


        #print_coords([(p.x, p.y) for p in done] + [(p.x, p.y) for p in rock])

        down = True
        for p in rock:
            if (p + Point.of(0, -1)) in done:
                down = False
                break

        if not down:
            for p in rock:
                maxy = max(maxy, p.y)
                done.add(p)

            #print(rock)
            break

        for p in rock:
            p.y -= 1

    it += 1

prints(maxy+extra)
