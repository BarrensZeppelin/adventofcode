#!/usr/bin/env pypy3
# 53/4

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

ins, adjs = sys.stdin.read().split("\n\n")

ADJ = {}
for l in lines(adjs):
    k, r = l.split(" = ")
    l, r = r.split(", ")
    ADJ[k] = (l[1:], r[:-1])
    print(k, ADJ[k])


res = 0

def path(start):
    r = 0
    cur = start
    for action in cycle(ins):
        if cur.endswith("Z"):
            break

        r += 1
        cur = ADJ[cur][action == "R"]
    return r

ans = [path(start) for start in ADJ if start.endswith("A")]
prints(math.lcm(*ans))
#
# cur = "AAA"
# for action in cycle(ins):
#     if cur == "ZZZ":
#         break
#
#     res += 1
#     cur = ADJ[cur][action == "R"]
#
# prints(res)
