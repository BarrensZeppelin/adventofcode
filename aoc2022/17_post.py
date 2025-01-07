#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

shapes = [
    set(
        Point.of(x, y)
        for y, l in enumerate(lines(s)[::-1])
        for x, c in enumerate(l)
        if c != "."
    )
    for s in """####

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
##""".split(
        "\n\n"
    )
]

directions = input()

for part, iterations in enumerate((2022, 1000000000000), start=1):
    maxy = shape_i = dir_i = it = extra = 0

    def get_shape(it=cycle(shapes)):
        global shape_i
        shape_i = (shape_i + 1) % len(shapes)
        return next(it)

    def get_direction(it=cycle(directions)):
        global dir_i
        dir_i = (dir_i + 1) % len(directions)
        return next(it)

    settled = {Point.of(x, 0) for x in range(7)}
    foundcyc = False
    memo = {}
    while it < iterations:
        if not foundcyc:
            # Number of top rows to include as key in cycle detection routine.
            # 10 did not work on my input.
            NUMROWS = 20
            k = (
                shape_i,
                dir_i,
                frozenset(
                    (p.x, p.y - maxy) for p in settled if (maxy - p.y) <= NUMROWS
                ),
            )
            # Check for a repeating state
            if k in memo:
                pit, pmaxy = memo[k]

                cyclen = it - pit
                remcyc = (iterations - it) // cyclen

                it += remcyc * cyclen
                extra = (maxy - pmaxy) * remcyc
                foundcyc = True

            memo[k] = (it, maxy)

        it += 1

        rock = [Point.of(p.x + 2, p.y + maxy + 4) for p in get_shape()]

        while True:
            d = Point.of(-1 if get_direction() == "<" else 1, 0)

            nrock = [p + d for p in rock]
            if all(0 <= p.x < 7 for p in nrock) and not settled & set(nrock):
                rock = nrock

            nrock = [p + Point.of(0, -1) for p in rock]
            if not settled & set(nrock):
                rock = nrock
            else:
                maxy = max(maxy, max(p.y for p in rock))
                settled.update(rock)
                break

    print(f"Part {part}: {maxy + extra}")
