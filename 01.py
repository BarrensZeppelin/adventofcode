#! /usr/bin/env pypy3
# 68/498
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

L = lines()

s = 0
best = 0
S = []
for l in L:
    if l == "":
        best = max(best, s)
        S.append(s)
        s = 0
    else:
        s += int(l)
best = max(best, s)
S.append(s)
print(best)

prints(sum(sorted(S)[-3:]))
