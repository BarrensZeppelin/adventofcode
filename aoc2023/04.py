#!/usr/bin/env pypy3
# 4/702

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

ls = lines()
N = len(ls)

cards = []
for id, l in enumerate(ls, start=1):
    _, rest = l.split(": ")
    winning, mine = rest.split(" | ")
    winning = set(ints(winning))
    r = 0
    for x in ints(mine):
        if x in winning:
            if r == 0:
                r = 1
            else:
                r += 1
                # r *= 2
    cards.append((id, r))

copies = [Counter() for _ in range(N)]
for i in range(N):
    copies[i][cards[i][0]] += 1


for i, (id, r) in enumerate(cards):
    if not r:
        continue

    print(id, copies[i])
    if r:
        win = 0
        for _, cnt in copies[i].items():
            win += cnt


        for j in range(id+1, id+r+1):
            print(id, r, j)
            # print(j, N, win)
            copies[j-1][j] += win



prints(sum(d for c in copies for d in c.values()))

# prints(res)
