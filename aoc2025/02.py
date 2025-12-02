#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

replace_stdin()

res = 0
S = input().strip().split(',')

for rng in S:
    a, b = map(int, rng.split('-'))
    for x in range(a, b+1):
        s = str(x)
        # if len(s) % 2:
        #     continue

        l = len(s) // 2
        for i in range(1, l+1):
            if len(s) % i:
                continue

            q = len(s) // i
            if s == s[:i] * q:
                res += x
                break
        # if s == s[:l] * 2:
        #     res += x


submit(res)
