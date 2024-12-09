#!/usr/bin/env pypy3
from __future__ import annotations

from itertools import chain, islice

from util import *

replace_stdin()

res = 0
L = [*map(int, input())]
N = sum(L)
files, free = [], []
i = 0
for j, d in enumerate(L):
    if j % 2:
        free.append((i, d))
    else:
        files.append((i, j // 2, d))
    i += d


def checksum(R):
    return sum(i * id for i, id in enumerate(R) if id > 0)


def part1():
    f = (p for j, d in free for p in range(j, j + d) if p < i)
    R = [-1] * N
    for i, id, d in reversed(files):
        for p in islice(chain(f, range(i, i + d)), d):
            R[p] = id

    return R


def part2():
    R = [-1] * N
    for i, id, d in reversed(files):
        for fi, (j, d2) in enumerate(free):
            if j >= i:
                R[i : i + d] = [id] * d
                break
            if d2 >= d:
                R[j : j + d] = [id] * d
                free[fi] = (j + d, d2 - d)
                break
    return R


print(f"Part 1: {checksum(part1())}")
print(f"Part 2: {checksum(part2())}")
