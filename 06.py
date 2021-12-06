#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))

C = Counter(ints())

for _ in range(256):
    NC = Counter()
    for k, v in C.items():
        if k == 0:
            NC[6] += v
            NC[8] += v
        else:
            NC[k-1] += v

    C = NC


print(sum(C.values()))
