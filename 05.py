#!/usr/bin/env pypy3
# 375/220
from __future__ import annotations

from functools import cmp_to_key

from util import *

replace_stdin()

res = 0


A, B = sys.stdin.read().split("\n\n")

adj = defaultdict(list)
for l in lines(A):
    a, b = ints(l)
    adj[a].append(b)

# order, x = topsort(adj)
# assert not x
#
# no = {k: i for i, k in enumerate(order)}


def key(a, b):
    if b in adj[a]:
        return -1
    elif a in adj[b]:
        return 1
    else:
        return 0


for l in lines(B):
    l = ints(l)
    for a, b in zip(l, l[1:]):
        if a in adj[b]:
            y = sorted(l, key=cmp_to_key(key))
            res += y[len(y) // 2]
            break
    # else:
    # res += l[len(l)//2]


submit(res)
