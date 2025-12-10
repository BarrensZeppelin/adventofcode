#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

import z4 as z3

res = 0

for l in lines():
    dest, *buttons, dc = l.split(' ')
    dc = ints(dc)
    buttons = [ints(b) for b in buttons]
    dest = dest[1:-1]
    bd = 0
    for i, c in enumerate(dest):
        if c == "#":
            bd |= 1 << i

    o = z3.Optimize()
    bs = z3.IntVector("b", len(buttons))
    ds = [0] * len(dc)
    for bx, b in zip(bs, buttons):
        o.add(bx >= 0)

        for i in b:
            ds[i] += bx

    for dx, d in zip(ds, dc):
        o.add(dx == d)

    obj = o.minimize(sum(bs))
    assert o.check() == z3.sat
    res += (obj.value().as_long())
    continue


    # def adj(bs: int, buttons=buttons):
    #     for b in buttons:
    #         nbs = bs
    #         for i in b:
    #             nbs ^= 1 << i
    #         yield nbs, 1

    def adj(bs: tuple[int, ...], buttons=buttons):
        for b in buttons:
            ncs = list(bs)
            for i in b:
                ncs[i] += 1
            if all(v <= dc[i] for i, v in enumerate(ncs)):
                yield tuple(ncs), 1

    # D, _ = dijkstra(adj, 0, target=bd)
    D, _ = dijkstra(adj, tuple([0] * len(dc)), target=tuple(dc))
    res += D[tuple(dc)]


submit(res)
