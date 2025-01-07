#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


monkeys = sys.stdin.read().split("\n\n")

for part, rounds in enumerate((20, 10**4), start=1):
    mon = []
    reduce = 1
    for i, monk in enumerate(monkeys):
        _, st, ops, ts, ifts, iffs = lines(monk)

        mon.append(
            (
                [*ints(st)],
                eval("lambda old: " + ops.split("new = ")[1]),
                next(ints(ts)),
                next(ints(ifts)),
                next(ints(iffs)),
            )
        )
        reduce *= mon[-1][2]
        # print(mon[-1])

    throws = [0] * len(mon)
    for rnd in range(rounds):
        for i, (its, op, t, ift, iff) in enumerate(mon):
            for worry in map(op, its):
                throws[i] += 1
                if part == 1: worry //= 3
                else: worry %= reduce
                mon[ift if worry % t == 0 else iff][0].append(worry)

            its.clear()

    a, b = sorted(throws)[-2:]
    print(a * b)
