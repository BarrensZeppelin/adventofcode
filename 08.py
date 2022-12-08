#! /usr/bin/env pypy3
# 20/5
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0

L = [list(map(int, l)) for l in lines()]
H = len(L)
W = len(L[0])
print(H, W)

for y, l in enumerate(L):
    for x, h in enumerate(l):
        ok = False
        seen = []
        for dy, dx in DIR:
            cx, cy = x + dx, y + dy
            i = 0
            while 0 <= cx < W and 0 <= cy < H:
                i += 1
                if L[cy][cx] >= h: break
                cx += dx; cy += dy
            else:
                ok = True
                #break

            seen.append(i)

        #res += ok

        z = 1
        for v in seen:
            z *= v

        res = max(res, z)


prints(res)
