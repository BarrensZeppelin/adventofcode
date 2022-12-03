#! /usr/bin/env pypy3
# 4/3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0

S = "ABC"
S2 = "XYZ"

for l in lines():
    a, b = l.split()


    i, j = S.index(a), S2.index(b)

    if j == 0:
        j = (i-1) % 3
    elif j == 1:
        j = i
    else:
        j = (i +1) % 3

    res += j+1
    if i == j:
        res += 3
    elif j == (i+1)%3:
        res += 6


prints(res)
