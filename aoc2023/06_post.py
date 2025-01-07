#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

T, D = (list(ints(input())) for _ in range(2))
for part in range(1, 3):
    if part == 2:
        T = [int("".join(map(str, T)))]
        D = [int("".join(map(str, D)))]

    res = 1
    for time, dist in zip(T, D):
        hi = time // 2
        assert hi * (time-hi) > dist
        p = binary_search(lambda x: x * (time-x) > dist, 1, hi)
        res *= time - 2*p + 1

    print(f"Part {part}: {res}")
