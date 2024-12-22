#!/usr/bin/env pypy3
# 130/290
from __future__ import annotations

from util import *

replace_stdin()

res = 0

def mix(x, y):
    return x ^ y

def prune(x):
    return x % 16777216

M = 16777216
c = Counter()
for l in lines():
    ix = x = int(l)
    # x = 123
    xs = [x]
    for _ in range(2000):
        x = mix(x, x*64) % M
        x = mix(x, x//32) % M
        x = mix(x, x*2048) % M
        xs.append(x)

    # res += x
    xs = [x % 10 for x in xs]
    # print(xs[:10])
    d = [b - a for a, b in zip(xs, xs[1:])]
    # print(xs, d)
    b = Counter()
    for i, t in enumerate(windows(d, 4)):
        t = tuple(t)
        p = xs[i + 4]
        if t not in b:
            b[t] = p

    # print(b[(0, 0, 0, 8)])
    for t, p in b.items():
        c[t] += p
    assert len(d) == 2000

res = max(c.values())
# res = max(c.items(), key=lambda t: t[1])

submit(res)
