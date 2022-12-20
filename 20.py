#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


res = 0

key = 811589153
L = list((i, v * key) for i, v in enumerate(ints()))
order = list(L)

for _ in range(10):
    for p in order:
        i = L.index(p)
        L.pop(i)
        j = (i + p[1]) % len(L)
        L.insert(j, p)

p = [p for p in order if p[1] == 0][0]
i = L.index(p)
N = len(L)
for x in range(1, 4):
    res += L[(i+x*1000)%N][1]

prints(res)
