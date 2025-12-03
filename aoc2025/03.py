#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

replace_stdin()

res = 0

for l in lines():
    xs = list(map(int, l))

    n = len(xs)
    @cache
    def f(i, t):
        if i == n:
            if t == 0:
                return 0
            return -(1 << 60)
        elif t == 0:
            return 0

        b = f(i+1, t)
        if t:
            v = xs[i]
            b = max(b, v * 10 ** (t-1) + f(i+1, t-1))
        # print(l, i, t, b)
        return b

    v = f(0, 12)
    print(l, v)
    # 171164621977817
    res += v
    continue

    best = 0
    for i, v in enumerate(xs):
        x = 10 * v + max(xs[i+1:], default=-1000)
        best = max(x, best)

    res += best


submit(res)
