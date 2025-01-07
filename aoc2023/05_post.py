#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

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
    for mapping_ranges in maps:
        next_intervals = []

        for l, r in intervals:
            for dest_start, source_start, length in mapping_ranges:
                if (cut := cut_interval(l, r, source_start, source_start + length)):
                    shift = dest_start - source_start
                    cl, (ml, mr), cr = cut
                    next_intervals.append((ml + shift, mr + shift))
                    intervals.extend(i for i in (cl, cr) if i)
                    break
            else:
                next_intervals.append((l, r))

        intervals = merge_intervals(next_intervals)

    print(f"Part {i}: {min(l for l, _ in intervals)}")
