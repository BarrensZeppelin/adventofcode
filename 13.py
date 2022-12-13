#! /usr/bin/env pypy3
# 32/30
from itertools import zip_longest
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

res = 0

def compare(a, b):
    if isinstance(a, list):
        if not isinstance(b, list):
            return compare(a, [b])

        for x, y in zip_longest(a, b, fillvalue=None):
            if x is None: return -1
            if y is None: return 1
            r = compare(x, y)
            if r != 0:
                return r

        return 0
    elif isinstance(b, list) and not isinstance(a, list):
        return compare([a], b)

    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


"""
for i, pas in enumerate(ps):
    a, b = map(eval, lines(pas))
    if compare(a, b) <= 0:
        print(i+1)
        res += i + 1
"""
class Comp:
    def __init__(self, x):
        self.x = x

    def __lt__(s, o):
        return compare(s.x, o.x) == -1

ls = [eval(l) for l in lines() if l]
ls.extend(([[2]], [[6]]))
ls.sort(key=Comp)

r = 1
for i, l in enumerate(ls):
    if l == [[2]] or l == [[6]]:
        r *= i+1

prints(r)
