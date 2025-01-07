#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))

#Y = 10
Y = 2000000

MAX = 4000000
#MAX = 20
MAXM = 4000000
beacons = set()
constraints = defaultdict(list)
for sx, sy, bx, by in map(ints, lines()):
    if by == Y: beacons.add(bx)

    dist = abs(by - sy) + abs(bx - sx)
    for ry in range(max(0, sy-dist), min(sy+dist+1, MAX+1)):
        rem = dist - abs(ry-sy)
        constraints[ry].append((sx-rem, sx+rem))

L = sorted(constraints[Y])
i = L[0][0]
j = i
part1 = 0
for x1, x2 in L:
    j = max(x1, j)
    if x2 >= j:
        part1 += x2 - j + 1
        j = x2+1

print(part1 - sum(any(x1 <= x <= x2 for x1, x2 in L) for x in beacons))

for y in range(MAX+1):
    done = 0
    for x1, x2 in sorted(constraints[y]):
        if done < x1:
            prints(done * MAXM + y)
            exit()

        done = max(done, x2+1)
        if done > MAX: break

assert False

