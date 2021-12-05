#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))

L = list(ints())


prev = 0
r = 0

for j, i in enumerate(L):
    i = sum(L[j:j+3])
    if i > prev:
        r += 1

    prev = i

print(r-1)
