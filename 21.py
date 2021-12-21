#! /usr/bin/env pypy3
# 3/19
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))


_, a, _, b = ints()

@lru_cache(maxsize=None)
def f(a, b, s1, s2, my_turn):

    res = 0
    tot = 0
    for roll in product(range(1, 4), repeat=3):
        na = (a + sum(roll) - 1) % 10 + 1

        if s1 + na >= 21:
            res += my_turn
            tot += 1
        else:
            x, y = f(b, na, s2, s1 + na, not my_turn)
            res += x
            tot += y

    return res, tot


w1, total = f(a, b, 0, 0, True)
print(max(w1, total - w1))

rolls = 0
s1 = s2 = 0

die = cycle(range(1, 101))

while True:
    rolls += 3
    r = sum(next(die) for _ in range(3))
    a = (a + r - 1) % 10 + 1
    s1 += a

    if s1 >= 1000:
        break

    a, b, s1, s2 = b, a, s2, s1

print(s2 * rolls)
