#! /usr/bin/env pypy3
from functools import reduce

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

alg, _, *I = lines()

D = {(x, y): c == "#" for y, l in enumerate(I) for x, c in enumerate(l)}

H, W = len(I), len(I[0])

for time in range(50):
    D = {
        (x, y): alg[
            reduce(
                lambda acc, v: acc * 2 + v,
                (
                    D.get((x + dx, y + dy), time % 2)
                    for dy in range(-1, 2)
                    for dx in range(-1, 2)
                ),
            )
        ]
        == "#"
        for y in range(-time - 1, H + time + 1)
        for x in range(-time - 1, W + time + 1)
    }

    if time in (1, 49):
        prints(sum(D.values()))
