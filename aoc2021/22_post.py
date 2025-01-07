#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


L = 50
cub = []
D = {}
xs, ys, zs = ([10 ** 10] for _ in range(3))
for l in lines():
    x1, x2, y1, y2, z1, z2 = ints(l)
    on = int(l.startswith("on"))

    xs += x1, x2 + 1
    ys += y1, y2 + 1
    zs += z1, z2 + 1

    cub.append((x1, x2, y1, y2, z1, z2, on))

    for x in range(max(x1, -L), min(x2, L) + 1):
        for y in range(max(y1, -L), min(y2, L) + 1):
            for z in range(max(z1, -L), min(z2, L) + 1):
                D[(x, y, z)] = on

L = []
for l in (xs, ys, zs):
    l[:] = sorted(set(l))
    L.append(len(l))

X, Y, Z = L

print(sum(D.values()))

# From PyRival
class BitArray:
    """implements bitarray using bytearray"""

    def __init__(self, size):
        self.bytes = bytearray((size >> 3) + 1)

    def __len__(self):
        return len(self.bytes) * 8

    def __getitem__(self, index):
        return (self.bytes[index >> 3] >> (index & 7)) & 1

    def __setitem__(self, index, value):
        if value:
            self.bytes[index >> 3] |= 1 << (index & 7)
        else:
            self.bytes[index >> 3] &= ~(1 << (index & 7))


C = [[BitArray(Z) for _ in range(Y)] for _ in range(X)]

from bisect import *

from tqdm import tqdm

lmask = 2 ** 8 - 1

for *me, on in tqdm(cub):
    xb, yb, zb = (
        (bisect_left(l, a), bisect_right(l, b))
        for a, b, l in zip(
            me[::2],
            me[1::2],
            (xs, ys, zs),
        )
    )
    z1, z2 = zb

    for x in range(*xb):
        for y in range(*yb):
            b1 = (z1 & ~7) + 8
            for z in range(z1, min(b1, z2)):
                C[x][y][z] = on

            while b1 + 8 <= z2:
                C[x][y].bytes[b1 // 8] = lmask * on
                b1 += 8

            for z in range(b1, z2):
                C[x][y][z] = on


idx = [[]]
for bs in range(1, 2 ** 8):
    nl = [i + 1 for i in idx[bs >> 1]]
    if bs & 1:
        nl.append(0)
    idx.append(nl)


res = 0
for x in range(X-1):
    xdiff = (xs[x + 1] - xs[x])
    for y in range(Y-1):
        area = xdiff * (ys[y + 1] - ys[y])
        for zb, b in enumerate(C[x][y].bytes):
            zb *= 8
            for i in idx[b]:
                res += area * (zs[zb + i + 1] - zs[zb + i])

prints(res)
