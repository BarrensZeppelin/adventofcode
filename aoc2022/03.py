#! /usr/bin/env pypy3
# 132/54
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0


L = lines()
for t in tile(L, 3):
    for c in set(t[0]) & set(t[1]) & set(t[2]):
        if c.isupper():
            res += ord(c) - ord('A') + 26+1
        else:
            res += ord(c) - ord('a') + 1


prints(res)
