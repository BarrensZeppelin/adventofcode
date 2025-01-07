#!/usr/bin/env pypy3
# 67/44
from __future__ import annotations

from util import *

replace_stdin()

res = 0

M, M2 = sys.stdin.read().split("\n\n")

INIT = lines(M)
R = dict(zip("#O.@", "## [] .. @.".split()))
NEW = ["".join(R[c] for c in l) for l in INIT]
G = Grid(NEW)

M2 = "".join(lines(M2))
D = dict(zip(">^<v", map(Point, DIR_NORTHNEG)))

sp = G.rev["@"][0]
G[sp.y][sp.x] = "."

# I ended up not needing this...
id = 0
crate_t = {}
for p in G.rev["["]:
    crate_t[p] = id
    id += 1

for c in M2:
    d = D[c]
    np = sp + d
    while G(np) in "[]":
        np += d
    if G(np) == "#":
        continue

    assert G(np) == "."
    dp = sp + d
    if d.y == 0:
        if d.x < 0:
            G[sp.y][np.x:sp.x+1] = G[sp.y][np.x+1:sp.x+1] + ["."]
        else:
            G[sp.y][sp.x:np.x+1] = ["."] + G[sp.y][sp.x:np.x]

        sp += d
    else:
        np = sp + d
        if G(np) == ".":
            sp = np
            continue

        pushing = {sp}
        dist = 0
        ok = True
        L = []
        while True:
            npush = set()
            for p in pushing:
                np = p + d
                if G(np) == "#":
                    ok = False
                    break
                if G(np) == "[":
                    npush.add(np)
                    npush.add(np + (1, 0))
                if G(np) == "]":
                    npush.add(np)
                    npush.add(np - (1, 0))
            else:
                dist += 1
                pushing = npush
                L.append(pushing)

            if not ok:
                break

            if not pushing:
                break

        if not ok:
            continue

        L.reverse()
        for l in L[1:]:
            # print([G(p) for p in l])
            # assert all(G(p) in "[]" for p in l)
            for p in l:
                c = G[p.y][p.x]
                assert c in "[]"
                np = p + d
                assert G(np) == "."
                G[np.y][np.x] = c
                G[p.y][p.x] = "."

        sp += d

for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c == "[":
            res += x + 100 * y

submit(res)
