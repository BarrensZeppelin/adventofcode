#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

res = 0

rngs, ings = sys.stdin.read().split("\n\n")

rngs = [ints(l) for l in lines(rngs)]

rngs = [(a, -b) for a, b in rngs]

merged = merge_intervals(rngs)
for a, b in merged:
    res += b - a + 1

# for ing in ints(ings):
#     for a, b in rngs:
#         b = -b
#         if a <= ing <= b:
#             res += 1
#             break

submit(res)
