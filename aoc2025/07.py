#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = Grid(lines())
s = G.rev["S"][0]
Q = [s]
R = defaultdict(int)
R[s] = 1
for p in Q:
    np = p + (0, 1)
    r = R[p]
    if not G.inbounds(np):
        res += r
        continue

    if G(np) in ".|":
        R[np] += r
        if G(np) == ".":
            G.set(np, "|")
            Q.append(np)
    elif G(np) == "^":
        # assert G(np) == "^", G(np)
        # res += 1
        for np in (np - (1, 0), np + (1, 0)):
            if G(np) in ".|":
                R[np] += r
                if G(np) == ".":
                    G.set(np, "|")
                    Q.append(np)


submit(res)
