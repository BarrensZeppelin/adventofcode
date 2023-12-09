#!/usr/bin/env pypy3
# 116/75

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

def f(ls):
    diffs = [ls[i] - ls[i-1] for i in range(1, len(ls))]
    if set(diffs) == {0}:
        return 0

    return diffs[0] - f(diffs)


for l in lines():
    ls = list(ints(l))
    v = f(ls)
    print(v)
    res += ls[0] - v

prints(res)
