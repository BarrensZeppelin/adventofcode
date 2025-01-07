#! /usr/bin/env pypy3
# 56/41
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


res = 0

map = []
BP = dict(zip(">v<^", [Point.of(*d) for d in DIR]))

for l in lines():
    map.append(l)

for x, c in enumerate(map[0]):
    if c == ".":
        start = (x, 0)
        break

H = len(map)
W = len(map[0])

for x, c in enumerate(map[-1]):
    if c == ".":
        end = (x, H-1)
        break


def blizz(time, x, y):
    # col
    ch = H - 2 + (map[0][x] == ".") + (map[-1][x] == ".")
    for by in range(H):
        if map[by][x] == "v":
            dist = y - by
            if dist % ch == time % ch:
                return True
        if map[by][x] == "^":
            dist = by - y
            if dist % ch == time % ch:
                return True

    ch = W - 2
    for bx in range(W):
        if map[y][bx] == ">":
            dist = x - bx
            if dist % ch == time % ch:
                return True
        if map[y][bx] == "<":
            dist = bx - x
            if dist % ch == time % ch:
                return True

    return False


Q = [(0, start, 0)]
V = {(0, start, 0)}
for i, p, state in Q:
    if p == end and state == 2:
        prints(i)
        exit()

    """
    mapp = {(x, y): c for y, l in enumerate(map) for x, c in enumerate(l)}
    mapp[p] = "E"
    print(i)
    print_coords(mapp)
    print()
    """

    x, y = p
    for dx, dy in DIR + ((0, 0),):
        nx = x + dx
        ny = y + dy

        if 0 <= nx < W and 0 <= ny < H:
            nst = state
            if state == 0 and (nx, ny) == end:
                nst = 1
            elif state == 1 and (nx, ny) == start:
                nst = 2

            nt = (i+1, (nx, ny), nst)
            if map[ny][nx] != "#" and nt not in V and not blizz(i+1, nx, ny):
                V.add(nt)
                Q.append(nt)

prints(res)
