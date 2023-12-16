#!/usr/bin/env pypy3
# 109/116

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

G = lines()
AA = {
  "|": [(0, -1), (0, 1)],
  "-": [(1, 0), (-1, 0)],
}

res = 0
DI = DIR_NORTHNEG
start = Point.of(0, 0)

W, H = len(G[0]), len(G)
def count(start, sd):
    V = {(start, sd)}
    Q = [(start, sd)]

    def nxt(np, nd):
        if (np, nd) in V:
            return
        V.add((np, nd))
        Q.append((np, nd))

    for p, d in Q:
        di = DI[d]
        np = p + di
        if not 0 <= np.x < W or not 0 <= np.y < H:
            continue

        c = G[np.y][np.x]
        if c in AA:
            adjs = AA[c]
            if di not in AA[c]:
                nxt(np, (d + 1) % 4)
                nxt(np, (d - 1) % 4)
            else:
                nxt(np, d)
        elif c == ".":
            nxt(np, d)
        elif c == "/":
            if d == 0:
                nxt(np, (d + 1) % 4)
            elif d == 1:
                nxt(np, (d - 1) % 4)
            elif d == 2:
                nxt(np, (d + 1) % 4)
            elif d == 3:
                nxt(np, (d - 1) % 4)
        elif c == "\\":
            if d == 0:
                nxt(np, (d - 1) % 4)
            elif d == 1:
                nxt(np, (d + 1) % 4)
            elif d == 2:
                nxt(np, (d - 1) % 4)
            elif d == 3:
                nxt(np, (d + 1) % 4)
        else:
            assert False

    ps = set(p for p, d in V)
    # print_coords(ps, ".")
    return len(ps)-1

# print(count(Point.of(3, -1), 3))
# exit()

best = 0
for y in range(len(G)):
    best = max(best, count(Point.of(-1, y), 0))
    best = max(best, count(Point.of(W, y), 2))

for x in range(W):
    best = max(best, count(Point.of(x, -1), 3))
    best = max(best, count(Point.of(x, H), 1))

# prints(count(Point.of(-1, 0), 0))
prints(best)
