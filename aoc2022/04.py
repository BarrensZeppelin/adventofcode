#! /usr/bin/env pypy3
# 28/810
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0
re2 = 0
for l in lines():
    a, b = l.split(',')
    l1, r1 = map(int, a.split('-'))
    l2, r2 = map(int, b.split('-'))


    ok = False
    for i in range(l1, r1+1):
        ok |= l2 <= i <= r2
    for i in range(l2, r2+1):
        ok |= l1 <= i <= r1


    re2 += bool(ok)
    #print(a, b, ok, r2)
    res += (l1 <= l2 and r2 <= r1) or (l2 <= l1 and r1 <= r2)


prints(re2)
