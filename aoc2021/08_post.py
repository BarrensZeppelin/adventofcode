#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))


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

    def f(c):
        return 'abcdefg'.index(c)

    ins = [list(map(f, s)) for s in ins]

    res = 0
    for perm in permutations(range(7)):
        if all(tuple(sorted(perm[j] for j in i)) in nums for i in ins):
            for o in out:
                res *= 10

                x = tuple(sorted(perm[f(c)] for c in o))
                res += nums.index(x)

            break
    else: assert False

    r += res

prints(r)
