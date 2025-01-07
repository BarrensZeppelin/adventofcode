#!/usr/bin/env pypy3
# 100/41
from __future__ import annotations

from util import *

replace_stdin()

res = 0

for l in lines():
    b, *vs = ints(l)

    def f(v, i):
        if i == len(vs):
            return v == b

        return f(v + vs[i], i + 1) or f(v * vs[i], i + 1) or f(int(str(v) + str(vs[i])), i + 1)

    if f(vs[0], 1):
        res += b


submit(res)
