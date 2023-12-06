#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

times = list(ints(input()))
distances = list(ints(input()))
times = [int("".join(map(str, times)))]
distances = [int("".join(map(str, distances)))]
N = len(times)

res = 1

for time, dist in zip(times, distances):
    d = 0
    won = 0
    for hold in range(time+1):
        if (time - hold) * hold > dist:
            won += 1
        # print(hold, move, dist)
        d += hold+1
    print(won)
    res *= won



prints(res)
