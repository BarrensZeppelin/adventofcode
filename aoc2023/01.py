#!/usr/bin/env pypy3
# 1717/57

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

D = dict(zip("one, two, three, four, five, six, seven, eight, nine".split(", "), range(1, 10)))

for l in lines():
    digs = []
    for i, c in enumerate(l):
        if c.isdigit():
            digs.append(int(c))
        else:
            for k, v in D.items():
                if l[i:].startswith(k):
                    digs.append(v)
                    break
    # ls = list(map(int, (c for c in l if c.isdigit())))
    # if not ls: continue
    # print(ls)
    # assert ls[0] > 0 and ls[-1] > 0
    res += digs[0] * 10 + digs[-1]


prints(res)
