#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))


depth = pos = 0
d2 = 0

for l in lines():
    a, x = l.split()
    x = int(x)
    if a == 'forward':
        pos += x
        d2 += x * depth
    elif a == 'down':
        depth += x
    else:
        depth -= x


print(pos * depth)
print(pos * d2)

