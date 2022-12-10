#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))

res = 0
cycles = dict(zip("noop addx".split(), (1, 2)))
cyc = 0
X = 1
for comm, *args in map(str.split, lines()):
    for _ in range(cycles[comm]):
        cyc += 1
        i = cyc % 40
        print("#" if abs((i-1)%40 - X) <= 1 else " ", end="")
        if i == 20: res += cyc * X
        if i == 0: print()

    if comm == "addx":
        X += int(args[0])
print(res)
