#!/usr/bin/env pypy3
# 3/11

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

pats = sys.stdin.read().split("\n\n")

res = 0

for pat in pats:
    pat = lines(pat)
    opat = pat

    mul = 100
    prev = None
    for _ in range(2):
        H = len(pat)
        for y in range(1, H):
            a = pat[:y]
            b = pat[y:]
            if all(x == y for x, y in zip(a[::-1], b)):
                # res += mul * y
                assert prev is None
                prev = mul, y
                # nwork.add((y, mul))
                break


        mul = 1
        pat = rotate(pat, 3)

    pat = opat
    mul = 100
    prev = None
    for _ in range(2):
        H = len(pat)
        for y in range(1, H):
            a = pat[:y]
            b = pat[y:]
            wrongs = 0
            for x, aaa in zip(a[::-1], b):
                for z, w in zip(x, aaa):
                    wrongs += z != w
            if wrongs == 1:
                res += mul * y
                assert prev is None
                prev = mul, y
                # nwork.add((y, mul))
                break


        mul = 1
        pat = rotate(pat, 3)


prints(res)
