#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

res = 0

G = Grid(lines())

r = G.crev
Q = []
for p in r["@"]:
    cnt = 0
    for d in OCTDIR:
        np = p + d
        cnt += G.at(np, default=None) == "@"
    if cnt < 4:
        Q.append(p)

for p in Q:
    G.set(p, ".")
    for nd in OCTDIR:
        np1 = p + nd
        if G.at(np1, ".") != "@":
            continue
        cnt = 0
        for d in OCTDIR:
            np = np1 + d
            cnt += G.at(np, default=None) == "@"
        if cnt < 4:
            Q.append(np1)


res = len(set(Q))
print("\n".join(map(str, G)))

submit(res)
