#! /usr/bin/env pypy3
# 1/37
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

r = 0

adj = defaultdict(list)
for l in lines():
    a, b = l.split('-')
    adj[a].append(b)
    adj[b].append(a)


def f(i, v, b):
    global r

    if i.islower() and v[i]:
        if i == 'start' or b: return
        b = True

    if i == 'end':
        r += 1
        return

    v[i] += 1

    for j in adj[i]:
        f(j, v, b)

    v[i] -= 1


f('start', Counter(), True)
print(r)

r = 0
f('start', Counter(), False)
prints(r)
