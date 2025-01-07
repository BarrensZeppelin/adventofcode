#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

replace_stdin()

Gs = [Grid(s.split(",")) for s in ("789,456,123, 0A", " ^A,<v>")]

part1 = part2 = 0


@cache
def sequence_cost(s: str, i: int, gi: int = 0) -> int:
    if i == 0:
        return len(s)

    G = Gs[gi]
    sp = p = G.crev["A"][0]
    cost = 0
    for c in s:
        d = G.crev[c][0] - p
        xs = (">" if d.x > 0 else "<") * abs(d.x)
        ys = ("v" if d.y > 0 else "^") * abs(d.y)

        opts: list[str] = []
        if G(p + (d.x, 0)) != " ":
            opts.append(xs + ys + "A")
        if G(p + (0, d.y)) != " ":
            opts.append(ys + xs + "A")

        cost += min(sequence_cost(s, i - 1, 1) for s in opts)
        p += d
    assert p == sp
    return cost


for l in lines():
    code = int(l[:-1])
    part1 += sequence_cost(l, 3) * code
    part2 += sequence_cost(l, 26) * code


print(f"Part 1: {part1}\nPart 2: {part2}")
