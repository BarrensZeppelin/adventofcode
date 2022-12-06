#! /usr/bin/env pypy3
# 15/15
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

S = input()
for i in range(len(S)+1):
    if len(set(S[max(0, i-14):i])) == 14:
        prints(i)
        break

