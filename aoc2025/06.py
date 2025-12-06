#!/usr/bin/env -S uv run
from __future__ import annotations

from util import *

replace_stdin()

res = 0

mat = rotate(lines())

nums = []
for l in mat:
    if all(c == " " for c in l):
        continue
    nums.append(int(''.join(l[:-1])))
    op = l[-1]
    if op != " ":
        if op == "+":
            res += sum(nums)
        else:
            from math import prod
            res += prod(nums)
        nums = []


# print(*mat, sep="\n")
# exit()

# mat = rotate([l.split() for l in lines()])
#
# for *nums, op in mat:
#     nums = [*map(int, nums)]
#     if op == "+":
#         res += sum(nums)
#     else:
#         from math import prod
#         res += prod(nums)


submit(res)
