#!/usr/bin/env pypy3
# 49/85

from functools import cmp_to_key
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

cards = [l.split(" ") for l in lines()]

from string import digits
STRENGTH = digits[2:] + "AKQJT"[::-1]
STRENGTH = "J" + digits[2:] + "AKQT"[::-1]
print(STRENGTH)


@lru_cache(maxsize=None)
def typ(card):
    C = Counter(card)
    if len(C) == 1:
        return 10
    elif len(C) == 2:
        return 9 if max(C.values()) == 4 else 8

    m = max(C.values())
    if m == 3:
        return 7

    if sum(1 for cnt in C.values() if cnt == 2) == 2:
        return 6

    if m == 2:
        return 5
    return 4


def cmp(a, b):
    a, ca, _ = a
    b, cb, _ = b
    ta, tb = map(typ, (a, b))
    if ta != tb:
        return ta - tb

    for c1, c2 in zip(ca, cb):
        if c1 != c2:
            return STRENGTH.index(c1) - STRENGTH.index(c2)
    return 0
    assert False

def trans(card):
    card, bid = card

    poss = []
    for c in card:
        if c == "J":
            poss.append(list(STRENGTH))
        else:
            poss.append([c])

    return max((("".join(x), card, bid) for x in product(*poss)), key=cmp_to_key(cmp))

    return card, bid

cards = list(map(trans, cards))

cards.sort(key=cmp_to_key(cmp))
# print(cards)

for i, (_, _, bid) in enumerate(cards, start=1):
    res += i * int(bid)

prints(res)
