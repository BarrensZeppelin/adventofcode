#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

L = sys.stdin.read().split("\n\n")
*shapes, problem = L

shapes = [lines(s)[1:] for s in shapes]
S = len(shapes)
area = [0] * S

for si, s in enumerate(shapes):
    for l in s:
        for c in l:
            area[si] += c == "#"

res = 0

for problem in lines(problem):
    W, H, *counts = ints(problem)
    reg_area = W * H
    shape_area = sum(area[si] * c for si, c in enumerate(counts))
    if reg_area < shape_area:
        continue

    res += 1
    # print(reg_area - shape_area)
    # res += reg_area >= shape_area

submit(res)
