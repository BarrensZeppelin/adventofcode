#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

decrement = Counter[int]()
p1 = p2 = prefix_sum = 0

for id, l in enumerate(lines(), start=1):
    prefix_sum -= decrement[id]
    copies = prefix_sum+1
    p2 += copies

    winning, mine = map(ints, l.split(": ")[1].split(" | "))
    winning = set(winning)
    won = sum(x in winning for x in mine)
    prefix_sum += copies
    decrement[id+won+1] += copies

    if won: p1 += 2**(won-1)

print(f"Part 1: {p1}\nPart 2: {p2}")
