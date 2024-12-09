#!/usr/bin/env pypy3
# 286/243
from __future__ import annotations

from util import *

replace_stdin()

res = 0

L = list(map(int, input()))
R = [-1] * sum(L)
i = 0
L2 = []
for id, d in enumerate(L):
    if id % 2 == 0:
        L2.append((id // 2, d))
    else:
        L2.append((-1, d))

moved = set()
for i in reversed(range(len(L2))):
    id, d = L2[i]
    if id == -1:
        continue

    if id in moved:
        continue

    moved.add(id)

    for j, (id2, d2) in enumerate(L2):
        if id2 != -1:
            continue
        if j >= i:
            break

        if d <= d2:
            L2[i] = (-1, d)
            L2[j] = (id, d)
            if d < d2:
                L2[j+1:j+1] = [(-1, d2-d)]
            break

print(L2)
# R = []
# for i, d in enumerate(L):
#     if i % 2 == 0:
#         R.extend([i//2] * d)
#     else:
#         while d:
#             j = len(L) - 1
#             if i == j:
#                 break
#             if j % 2 == 1:
#                 L.pop()
#                 continue
#             d2 = L[j]
#             u = min(d, d2)
#             d -= u
#             L[j] -= u
#             R.extend([j//2] * u)
#             if not L[j]:
#                 L.pop()
#
# print(R)
i = 0
for id, d in L2:
    if id == -1:
        i += d
        continue

    print(id, d, i)
    for j in range(i, i+d):
        res = res + j * id

    i += d

submit(res)
