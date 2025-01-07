#!/usr/bin/env pypy3
from __future__ import annotations

import z4 as z3

from util import *

replace_stdin()

res = 0

A, B, C, *P = ints()
s = z3.Solver()
init_A = z3.BitVec("IA", 64)
s.add(init_A >= 0)


def check():
    r = s.check()
    assert r != z3.unknown, r
    return r == z3.sat


def g(v: int):
    with s:
        s.add(init_A <= v)
        return check()


def f(ip: int, j: int, A, B, C):
    while 0 <= ip < len(P):
        p, op = P[ip : ip + 2]
        cop = (0, 1, 2, 3, A, B, C, 0)[op]

        match p:
            case 0: A >>= cop
            case 1: B ^= op
            case 2: B = cop % 8
            case 4: B ^= C
            case 6: B = A >> cop
            case 7: C = A >> cop

            case 3:
                with s:
                    s.add(A != 0)
                    if check():
                        f(op, j, A, B, C)

                with s:
                    s.add(A == 0)
                    if check():
                        f(ip + 2, j, A, B, C)

                return

            case 5:
                if j < len(P):
                    with s:
                        s.add(cop % 8 == P[j])
                        if check():
                            f(ip + 2, j + 1, A, B, C)

                return

        ip += 2

    # Solver + binary search turns out to be several times faster than Optimize w. minimize
    prints(binary_search(g, 0))
    sys.exit(0)


f(0, 0, init_A, 0, 0)
