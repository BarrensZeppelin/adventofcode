#! /usr/bin/env pypy3
# 13/6
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

r = 0

pt = {')': 3, ']': 57, '}': 1197, '>': 25137}
op = {')': '(', ']': '[', '}': '{', '>': '<'}

score = ' )]}>'

sc = []

for l in lines():
    S = []
    for c in l:
        if c in op:
            if not S or S[-1] != op[c]:
                r += pt[c]
                break

            S.pop()
        else:
            S.append(c)

    else:
        x = 0
        while S:
            x *= 5
            c = S.pop()

            for k, v in op.items():
                if v == c:
                    x += score.index(k)
                    break
            else:
                assert False

        sc.append(x)

sc.sort()

print(r)
prints(sc[len(sc)//2])
