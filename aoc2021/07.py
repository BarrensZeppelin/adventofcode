#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))

V = list(ints())
V.sort()
n = len(V)//2

x = V[n//2]

# Using the "median" didn't work because I divided len(V) by 4...
print(min(sum(abs(y - x) for y in V) for x in V))

prints(min(sum(abs(y - x) * (abs(y - x) + 1) // 2 for y in V) for x in range(min(V), max(V)+1)))
