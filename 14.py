#! /usr/bin/env pypy3
# 42/82
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))


s = input()
input()

adj = {}

for l in sys.stdin:
    l = l.rstrip()
    a, b = l.split(" -> ")
    adj[a] = b

after = defaultdict(Counter)
for c1, c2 in zip(s, s[1:]):
    after[c1][c2] += 1

for _ in range(40):
    nafter = defaultdict(Counter)
    for c1, v in after.items():
        for c2, cnt in v.items():
            nafter[c1][adj[c1 + c2]] += cnt
            nafter[adj[c1 + c2]][c2] += cnt

    after = nafter

C = sum(after.values(), start=Counter())
C[s[0]] += 1

S = sorted(C.values())
prints(S[-1] - S[0])
