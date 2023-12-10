#!/usr/bin/env pypy3
# 17/107

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))


ADJ = {
 "|": [Point.of(0, 1), Point.of(0, -1)],
 "-": [Point.of(1, 0), Point.of(-1, 0)],
 "F": [Point.of(0, 1), Point.of(1, 0)],
 "J": [Point.of(0, -1), Point.of(-1, 0)],
 "L": [Point.of(1, 0), Point.of(0, -1)],
 "7": [Point.of(0, 1), Point.of(-1, 0)]
}

G = [list(l.replace("O", ".").replace("I", ".")) for l in lines()]
for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c == "S":
            start = Point.of(x, y)
            break

G[start.y][start.x] = "J"
# G[start.y][start.x] = "F"

H = len(G)
W = len(G[0])


NG = [["."] * (W*3) for _ in range(H*3)]
for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c == ".":
            continue
        NG[y*3+1][x*3+1] = c
        for diff in ADJ[c]:
            n = Point.of(3*x+1, 3*y+1) + diff
            NG[n.y][n.x] = "|" if diff.x == 0 else "-"

G = NG

# print("\n".join("".join(l) for l in NG))
# exit()


# for k, aa in ADJ.items():
#     for diff in aa:
#         n = start + diff
#         if G[n.y][n.x] == ".":
#             break
#     else:
#         print(k)

start = start * 3 + Point.of(1, 1)

D = {start: 0}
Q = [start]


for p in Q:
    for diff in ADJ[G[p.y][p.x]]:
        n = p + diff
        assert G[n.y][n.x] != "."
        if n not in D:
            D[n] = D[p] + 1
            Q.append(n)

# prints(max(D.values()))

in_loop = set(D)
H = len(G)
W = len(G[0])
V = set()
Q = []
for x in range(W):
    Q.extend([Point.of(x, -1), Point.of(x, H)])
for y in range(H):
    Q.extend([Point.of(-1, y), Point.of(W, y)])
V = set(Q)

for p in Q:
    assert p not in in_loop
    for diff in DIR:
        n = p + diff
        if 0 <= n.x < W and 0 <= n.y < H and n not in in_loop and n not in V:
            V.add(n)
            Q.append(n)

V |= in_loop

RG = [[] for _ in range(H//3)]
res = 0
for y in range(1, H, 3):
    for x in range(1, W, 3):
        p = Point.of(x, y)
        if p not in V:
            res += 1
            RG[y//3].append("I")
        elif p not in in_loop:
            RG[y//3].append("O")
        else:
            RG[y//3].append(G[y][x])

print("\n".join("".join(l) for l in RG))
prints(res)
