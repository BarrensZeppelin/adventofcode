#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

replace_stdin()

D = dict(zip(">^<v", map(Point, DIR_NORTHNEG), strict=True))

G, I = map(lines, sys.stdin.read().split("\n\n"))
I = "".join(I)


def solve(M: list[str]) -> int:
    G = Grid(M)
    sp = G.rev["@"][0]

    for d in map(D.__getitem__, I):

        def adj(p: Point[int]):
            if (c := G(p)) in "O[]":
                yield p + d
            if c == "[":
                yield p + (1, 0)
            if c == "]":
                yield p - (1, 0)

        push, _, _ = bfs(adj, sp + d)
        if any(G(p) == "#" for p in push):
            continue

        push = {p: G(p) for p in push if G(p) in "O[]"}
        for p in push:
            G[p.y][p.x] = "."
        for p, c in push.items():
            p += d
            G[p.y][p.x] = c

        sp += d

    return sum(p.x + 100 * p.y for p in G.points() if G(p) in "O[")


print(f"Part 1: {solve(G[:])}")

replace = dict(zip("#O.@", "## [] .. @.".split(), strict=True))
print(f"Part 2: {solve([''.join(replace[c] for c in l) for l in G])}")
