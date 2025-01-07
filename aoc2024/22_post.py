#!/usr/bin/env pypy3
from __future__ import annotations

from itertools import accumulate, pairwise

from util import *

replace_stdin()

M = 16777216


def f(x: int, _) -> int:
    x ^= x * 64
    x %= M
    x ^= x // 32
    x %= M
    x ^= x * 2048
    return x % M


p1 = 0
c = Counter[int]()
for x in ints():
    xs = list(accumulate(range(2000), f, initial=x))
    p1 += xs[-1]
    prices = [x % 10 for x in xs]
    diffs = [b - a for a, b in pairwise(prices)]
    c.update({tuple(s): p for s, p in zip(windows(diffs, 4)[::-1], prices[::-1], strict=False)})

print(f"Part 1: {p1}\nPart 2: {max(c.values())}")
