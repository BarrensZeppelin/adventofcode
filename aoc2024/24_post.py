#!/usr/bin/env pypy3
from __future__ import annotations

import random

from util import *

replace_stdin()

A, B = sys.stdin.read().split("\n\n")

ops: dict[str, list[str]] = {}
for l in lines(B):
    *op, _, dest = l.split()
    ops[dest] = op


def getv(s: str, x: int, y: int) -> int:
    if s[0] == "x":
        return (x >> int(s[1:])) & 1
    if s[0] == "y":
        return (y >> int(s[1:])) & 1
    av, op, bv = ops[s]
    a, b = getv(av, x, y), getv(bv, x, y)
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    assert False, op


x = y = 0
for l in lines(A):
    a, b = l.split(": ")
    if b == "1":
        i = 1 << int(a[1:])
        if a[0] == "x":
            x |= i
        else:
            y |= i

z = sum(getv(s, x, y) << int(s[1:]) for s in ops if s[0] == "z")
print(f"Part 1: {z}")


@cache
def testf(i: int):
    DIFF = 9
    tests = tile(random.choices(range(1 << i), k=2 << DIFF), 2)
    random.shuffle(tests)
    return tests


def mkadj():
    return defaultdict(list, {s: [a, b] for s, (a, _, b) in ops.items()})


def is_cyclic():
    return topsort(mkadj())[1]


def swappable(s: str):
    return set(bfs(mkadj(), s)[1]) & ops.keys()


def f(i: int, swapped: set[str]) -> None:
    if i == 46:
        assert len(swapped) == 8
        res = ",".join(sorted(swapped))
        print(f"Part 2: {res}")
        sys.exit()

    def check():
        for a, b in testf(i):
            exp = a + b
            for j in range(i + 1):
                if getv(f"z{j:02}", a, b) != (exp >> j) & 1:
                    return False

        return True

    if check():
        return f(i + 1, swapped)
    ind = " " * len(swapped)
    print(f"{ind}Found error at bit {i} ({len(swapped) // 2} swaps)")
    if len(swapped) == 8:
        return

    inside = swappable(f"z{i:02}") - swapped
    outside = set(ops) - inside - swapped
    good = 0
    for a, b in chain(product(inside, outside), combinations(inside, 2)):
        ops[a], ops[b] = ops[b], ops[a]
        if not is_cyclic() and check():
            good += 1
            print(f"{ind}{i}: {a} <-> {b} (try {good})")
            f(i, swapped | {a, b})
            print(f"{ind}{i}: Backtracking {a} <-> {b}")
        ops[a], ops[b] = ops[b], ops[a]


random.seed(2024)
f(0, set())
