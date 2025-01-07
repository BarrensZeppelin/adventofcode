#! /usr/bin/env python
from util import *

from functools import reduce
from z3 import And, BitVec, Distinct, Or, Solver


if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_z3.py", ".in"))

nums = [
    (0, 1, 2, 4, 5, 6),
    (2, 5),
    (0, 2, 3, 4, 6),
    (0, 2, 3, 5, 6),
    (1, 2, 3, 5),
    (0, 1, 3, 5, 6),
    (0, 1, 3, 4, 5, 6),
    (0, 2, 5),
    tuple(range(7)),
    (0, 1, 2, 3, 5, 6),
]

# Represent the segments as bitmasks
numb = [sum(1 << i for i in t) for t in nums]

# X represents the permutation, the values should therefore be distinct and
# range from 0 to 6
X = [BitVec(f"X__{i}", 7) for i in range(7)]
S = Solver()
S.add(Distinct(X), *[And(0 <= x, x < 7) for x in X])


def f(c):
    return "abcdefg".index(c)


r = 0
for l in lines():
    S.push()

    pat, e = l.split(" | ")

    ins = pat.split()
    out = e.split()

    for i, signal_ in enumerate(ins):
        signal = list(map(f, signal_))

        # Make a helper variable that is equal to the bitmask of the signal
        # after the permutation is applied.
        bv = BitVec(f"bv_{i}", 7)
        signal_shifts = [1 << X[v] for v in signal]
        S.add(bv == reduce(lambda expr, v: expr | v, signal_shifts))

        # The bitmask should be equal to one of the valid bitmasks.
        # We can save some time by only considering bitmasks with the same number of bits set.
        S.add(Or([bv == exp for t, exp in zip(nums, numb) if len(t) == len(signal)]))

    print(S.check())
    m = S.model()

    M = [m.evaluate(x).as_long() for x in X]

    res = 0
    for l in out:
        res *= 10

        v = sum(1 << M[i] for i in map(f, l))
        res += numb.index(v)

    S.pop()
    r += res

prints(r)
