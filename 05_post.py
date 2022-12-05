#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))

res = 0
stack, moves = sys.stdin.read().split('\n\n')
stacks_ = [" ".join(l).split() for l in rotate(lines(stack)[:-1], times=-1)[1::4]]

for part in (1, -1):
    stacks = [*map(list, stacks_)]
    for a, b, c in map(ints, lines(moves)):
        stacks[c-1] += [stacks[b-1].pop() for _ in range(a)][::part]

    print("".join(s.pop() for s in stacks))
