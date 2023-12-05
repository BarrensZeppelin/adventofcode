#!/usr/bin/env python3
import z4 as z3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_z3.py", ".in"))

seeds, *blocks = sys.stdin.read().split("\n\n")
seeds = list(ints(seeds))

maps = []
for block in blocks:
    ls = lines(block)[1:]
    maps.append([tuple(ints(l)) for l in ls])

for i, intervals in enumerate((
    [(seed, seed+1) for seed in seeds],
    [(seed, seed+len) for seed, len in tile(seeds, 2)],
), start=1):
    solver = z3.Optimize()
    prev = z3.Int("seed_0")
    solver.add(z3.Or(*[z3.And(l <= prev, prev < r) for l, r in intervals]))
    for j, mapping_ranges in enumerate(maps, start=1):
        next = z3.Int(f"seed_{j}")
        els = next == prev

        for dest_start, source_start, length in mapping_ranges:
            shift = dest_start - source_start
            els = z3.If(
                z3.And(source_start <= prev, prev < source_start + length),
                next == prev + shift,
                els
            )

        solver.add(els)
        prev = next

    solver.minimize(prev)
    assert solver.check() == z3.sat
    print(f"Part {i}: {solver.model().eval(prev)}")
