#! /usr/bin/env pypy3
# 97/41
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0
#Y = 10
Y = 2000000

bad = set()
beac = set()
MAX = 4000000
#MAX = 20
MAXM = 4000000
bads = defaultdict(list)
for l in lines():
    sx, sy, bx, by = ints(l)

    if by == Y: beac.add(bx)

    dy = abs(by - sy)
    dx = abs(bx - sx)
    dist = dy + dx
    """
    yd = abs(sy - Y)
    if yd > dist:
        continue
    """

    for offy in range(-dist, dist+1):
        ry = sy + offy
        x1 = sx - (dist - abs(offy))
        x2 = sx + (dist - abs(offy))
        bads[ry].append((x1, x2))
    #print(sx, sy, bx, by, dist)
    """
    for x in range(sx - (dist - yd), sx + (dist - yd)+1):
        #print(x)
        bad.add(x)
    """
    #print()

#prints(len(bad - beac))

for y in range(MAX+1):
    l = bads[y]
    l.sort()
    done = 0
    #print(l)
    for x1, x2 in l:
        if done < x1:
            print(done, y)
            prints(done * MAXM + y)
            exit()

        done = max(done, x2+1)
        if done > MAX: break

assert False

