#!/usr/bin/env pypy3
# 117/98
from __future__ import annotations

import z4 as z3
from util import *

replace_stdin()

res = 0
A, B = sys.stdin.read().split("\n\n")

G = dict()
for l in lines(A):
    a, b = l.split(": ")
    G[a] = int(b)

ops = {}
for l in lines(B):
    x, dest = l.split(" -> ")
    a, op, b = x.split()
    ops[dest] = (a, op, b)

zs = {s for s in ops if s[0] == "z"}
zs = sorted(zs, key=lambda x: int(x[1:]), reverse=True)

def sim(G):
    n = len(zs)

    i = 0
    while i < n:
        for d, (a, op, b) in ops.items():
            if d in G:
                continue
            if a in G and b in G:
                x, y = G[a], G[b]
                if op == "AND":
                    G[d] = x & y
                elif op == "OR":
                    G[d] = x | y
                elif op == "XOR":
                    G[d] = x ^ y
                else:
                    assert False

                if d in zs:
                    i += 1

    return int("".join(str(G[z]) for z in zs), 2)

import random

s = z3.Solver()
VAR = {s: z3.BitVec(s, 46) for s in list(G) + list(ops)}
for dest, (a, op, b) in ops.items():
    dv, av, bv = VAR[dest], VAR[a], VAR[b]
    if op == "AND":
        s.add(dv == (av & bv))
    elif op == "OR":
        s.add(dv == (av | bv))
    elif op == "XOR":
        s.add(dv == (av ^ bv))

xs = [f"x{i:02}" for i in range(45)]
ys = [f"y{i:02}" for i in range(45)]
# xs = {s for s in ops if s[0] == "x"}
# xs = sorted(xs, key=lambda x: int(x[1:]), reverse=True)
# ys = {s for s in ops if s[0] == "y"}
# ys = sorted(ys, key=lambda x: int(x[1:]), reverse=True)

@cache
def testf(i: int):
    DIFF = 6
    if i < DIFF:
        tests = list(product(range(1 << i), repeat=2))
    else:
        tests = []
        for _ in range(1 << (2*DIFF)):
            a = random.randrange(1 << i)
            b = random.randrange(1 << i)
            tests.append((a, b))

    random.shuffle(tests)
    return tests

def mkadj():
    adj = {s: [a, b] for s, (a, _, b) in ops.items()}
    for s in G:
        adj[s] = []
    return adj

def is_cyclic():
    return topsort(mkadj())[1]


def swappable(s: str):
    return set(bfs(mkadj(), s)[1]) - set(G)


def f(i: int, swapped: set[str]):
    if i == 46:
        res = ",".join(sorted(swapped))
        submit(res)
        exit()

    def getv(s: str, a: int, b: int) -> int:
        if s[0] == "x":
            return (a >> int(s[1:])) & 1
        if s[0] == "y":
            return (b >> int(s[1:])) & 1
        av, op, bv = ops[s]
        x, y = getv(av, a, b), getv(bv, a, b)
        if op == "AND":
            return x & y
        if op == "OR":
            return x | y
        if op == "XOR":
            return x ^ y

    def check():
        for a, b in testf(i):
            for j in range(i+1):
                x = getv(f"z{j:02}", a, b)
                if x != ((a + b) >> j) & 1:
                    return False


        return True

    works = check()
    print(i, works, swapped)
    if works:
        f(i+1, swapped)
        return

    if len(swapped) == 8:
        return

    inside = swappable(f"z{i:02}") - swapped
    outside = set(ops) - swapped
    to_test = list(product(inside, outside)) + list(combinations(inside, 2))
    random.shuffle(to_test)
    # can_swap = swappable(f"z{i:02}") - swapped
    # for a, b in combinations(can_swap, 2):
    for a, b in to_test:
        if a == b: continue
        ops[a], ops[b] = ops[b], ops[a]
        swapped.add(a)
        swapped.add(b)
        if not is_cyclic() and check():
            f(i, swapped)
        swapped.remove(a)
        swapped.remove(b)
        ops[a], ops[b] = ops[b], ops[a]

f(0, set())
exit()

correct = set()
# X, Y = z3.BitVecs("X Y", 46)
# Z = 0
# for i in range(46):
#     v = VAR[zs[~i]]
#     Z |= v << i
# print(Z)
#
# s.add(X >> 45 == 0)
# s.add(Y >> 45 == 0)
# for i in range(46):
#     with s:
#         c = (Z >> i) & 1 == ((X + Y) >> i) & 1
#         c = z3.Implies(z3.And(X >> 45 == 0, Y >> 45 == 0), c)
#         s.add(z3.ForAll([X, Y], c))
#         if s.check() == z3.sat:
#             correct.add(zs[~i])

# for _ in range(10**4):
#     # a = random.randrange(0, 1 << 44)
#     # b = random.randrange(0, 1 << 44)
#     # G = dict()
#     # for i, (x, y) in enumerate(zip(xs, ys)):
#     #     G[x] = (a >> i) & 1
#     #     G[y] = (b >> i) & 1
#
#     e = a + b
#     r = sim(G)
#     for i in range(45):
#         if (r >> i) & 1 != (e >> i) & 1:
#             correct.discard(zs[~i])


print(len(zs))
print(sorted(correct), len(correct))


print(len(ops))

# submit(res)
