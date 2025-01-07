#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

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

r = 0
for l in lines():
    pat, e = l.split(' | ')

    ins = pat.split()
    out = e.split()

    M = [-1] * 7

    def f(c):
        return 'abcdefg'.index(c)

    ins = [set(map(f, s)) for s in ins]

    def get(l):
        for s in ins:
            if len(s) == l:
                return s

    ones = get(2)
    sev = get(3)

    poss = [set(range(7)) for _ in range(7)]
    poss[0] = {(sev - ones).pop()}

    for i in range(1, 7): poss[i] -= poss[0]

    poss[2] = set(ones)
    poss[5] = set(ones)

    four = get(4)

    for p in ins:
        for v, req in enumerate(nums):
            if len(p) == len(req) and len(p) in (2, 3, 4, 7):
                for i in req:
                    poss[i] &= p

                for k, req2 in enumerate(nums):
                    if v != k and set(req) < set(req2):
                        for i in (set(req2) - set(req)):
                            poss[i] -= p

    # look for 9
    for p in ins:
        if len(p) != 6: continue
        if sev < p and four < p:
            rem = p - sev - four
            poss[6] = rem

            for i in range(1, 6):
                poss[i] -= rem

            break

    # look for 0
    for p in ins:
        if len(p) != 6: continue
        if (sev | poss[4] | poss[6]) < p:
            rem = p - sev - poss[4] - poss[6]
            assert len(rem) == 1, rem

            poss[1] = rem
            for i in range(7):
                if i != 1:
                    poss[i] -= rem
        #for v, req in enumerate(nums):

    rem = four - ones - poss[1]
    assert len(rem) == 1, rem
    poss[3] = rem
    for i in range(7):
        if i != 3:
            poss[i] -= rem

    # look for 5
    for p in ins:
        if len(p) != 5: continue

        if (poss[0] | poss[1] | poss[3] | poss[6]) < p:
            rem = p - (poss[0] | poss[1] | poss[3] | poss[6])
            poss[5] = rem
            poss[2] = ones - rem
            #rem -= ones

    M = [v.pop() for v in poss]
    rev = [-1] * 7
    for i, v in enumerate(M):
        rev[v] = i

    res = 0
    for l in out:
        res *= 10

        l = list(map(f, l))

        on = set(rev[v] for v in l)

        for i, v in enumerate(nums):
            if set(v) == on:
                res += i
                break
        else:
            assert False

    r += res

prints(r)
