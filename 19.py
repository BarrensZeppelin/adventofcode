#!/usr/bin/env pypy3
# 120/64
from __future__ import annotations

from util import *

replace_stdin()

res = 0

A, B = sys.stdin.read().split("\n\n")
pats = A.split(", ")

for l in lines(B):
    print(l, pats)
    @cache
    def f(i: int):
        if i == len(l):
            return 1

        r = 0
        for p in pats:
            if l[i:i+len(p)] == p and f(i+len(p)):
                r += f(i+len(p))

        return r

    res += f(0)


submit(res)
