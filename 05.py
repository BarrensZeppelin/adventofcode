#! /usr/bin/env pypy3
# 144/98
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0
stack, moves = sys.stdin.read().split('\n\n')

stacks = [[] for _ in range(10)]
for l in (lines(stack)[:-1]):
    for i in range(9):
        c = l[4*i+1]
        if c != " ":
            stacks[i+1].append(c)

for l in stacks:
    l.reverse()


for l in lines(moves):
    a, b, c = ints(l)
    ss = [stacks[b].pop() for _ in range(a)]
    ss.reverse()
    stacks[c] += ss
    """
    for _ in range(a):
        stacks[c].append(stacks[b].pop())
    """

prints("".join(s[-1] for s in stacks if s))
