#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('_post.py', '.in'))

adj = make_adj((l.split('-') for l in lines()), both=True)
idx = {name: i for i, name in enumerate(a for a in adj if a.islower())}

@lru_cache(maxsize=None)
def f(i: str, v: int, d: bool) -> int:
    if i == 'end':
        return 1

    r = 0
    for j in adj[i]:
        if j.isupper(): r += f(j, v, d)
        elif j != 'start':
            seen = v & (1 << idx[j]) != 0
            if not seen or not d:
                r += f(j, v | (1 << idx[j]), seen or d)

    return r


s = 'start'
print(f(s, 1 << idx[s], True))
prints(f(s, 1 << idx[s], False))


