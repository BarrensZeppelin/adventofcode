#! /usr/bin/env pypy3
from util import *

sys.stdin = open(__file__.replace('py', 'in'))
input = sys.stdin.readline

I = list(map(int, input().split(',')))

boards = []
M = defaultdict(list)

bs = 0
while True:
    _ = input()
    if _ == '': break

    B = [list(ints(input())) for _ in range(5)]
    boards.append(B)

    for c in range(5):
        s = {B[y][c] for y in range(5)}
        for i in s: M[i].append((s, bs))

    for y in range(5):
        s = {B[y][c] for c in range(5)}
        for i in s: M[i].append((s, bs))

    bs += 1


rem = set(range(bs))
D = set()
for i in I:
    D.add(i)

    for s, bi in M[i]:
        if i in s:
            s.remove(i)

            if not s:
                rem.discard(bi)

                if not rem:
                    r = sum(v for l in boards[bi] for v in l if v not in D)
                    print(r * i)
                    exit()

    """
    for s, bi in M[i]:
        assert i in s
        s.remove(i)

        if not s:
            r = sum(v for l in boards[bi] for v in l if v not in D)
            print(r * i)
            exit()
    """
