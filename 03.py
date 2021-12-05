#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))

g = e = 0

L = lines()
N = len(L[0])

def f(t):
    P = list(L)

    for x in range(N):
        C = Counter()
        for l in P:
            C[l[x]] += 1

        o, z = C['1'], C['0']

        if t:
            keep = '1' if o >= z else '0'
        else:
            keep = '0' if z <= o else '1'

        P = [l for l in P if l[x] == keep]
        if len(P) == 1:
            print('found', P[0])
            return int(P[0], 2)

        assert P

print(f(True) * f(False))
exit()

for x in range(N):
    C = Counter()
    for l in L:
        C[l[x]] += 1

    g <<= 1
    e <<= 1
    o, z = C['1'], C['0']
    if o > z:
        g += 1
    else:
        e += 1


print(g * e)
