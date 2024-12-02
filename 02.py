#!/usr/bin/env pypy3
# 50/115
from __future__ import annotations

from util import *

replace_stdin()

"""
L = sys.stdin.read().split("\n\n")

tile, rotate, Point, sign
"""

res = 0

def safe(X: list[int]):
    if sorted(X) == X or sorted(X, reverse=True) == X:
        for x, y in zip(X, X[1:]):
            if not 1 <= abs(x - y) <= 3:
                return False
        else:
            print(X)
            return True


for l in lines():
    X = ints(l)
    if safe(X) or any(safe(X[:i] + X[i + 1:]) for i in range(len(X))):
        res += 1

submit(res)
