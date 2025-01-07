#!/usr/bin/env pypy3
# 199/71
from __future__ import annotations

from util import *

replace_stdin()

res = 0

@cache
def f(c, i):
    if i == 0:
        return 1
    Y = []
    if c == 0:
        Y.append(1)
    elif len(str(c)) % 2 == 0:
        h = len(str(c)) // 2
        Y.append(int(str(c)[:h]))
        Y.append(int(str(c)[h:]))
    else:
        Y.append(c*2024)
    return sum(f(y, i-1) for y in Y)

X = ints()
# for _ in range(75):
#     Y = []
#     for c in X:
#         n = 0
#         if c == 0:
#             Y.append(1)
#         elif len(str(c)) % 2 == 0:
#             h = len(str(c)) // 2
#             Y.append(int(str(c)[:h]))
#             Y.append(int(str(c)[h:]))
#         else:
#             Y.append(c*2024)
#
#     X = Y


res = sum(f(x, 75) for x in X)
submit(res)
