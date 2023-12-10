#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


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

start_conn = []
for diff in map(Point, DIR):
    n = start + diff
    if -diff in ADJ.get(G[n.y][n.x], []):
        start_conn.append(diff)

start_sym = next(k for k, aa in ADJ.items() if set(start_conn) == set(aa))
assert start_sym == "J"

G[start.y][start.x] = start_sym

V = {start}
Q = [start]

for p in Q:
    for diff in ADJ[G[p.y][p.x]]:
        n = p + diff
        if n not in V:
            V.add(n)
            Q.append(n)
            break

print(f"Part 1: {len(Q)//2}")
# Shoelace formula
area = abs(sum(Q[i-1].cross(p) for i, p in enumerate(Q)))
# Pick's theorem
print(f"Part 2: {(area - len(Q) + 2) // 2}")
