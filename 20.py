#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

alg = input()
input()

I = [l.rstrip() for l in sys.stdin]

D = defaultdict(bool)
for y, l in enumerate(I):
    for x, c in enumerate(l):
        D[(x, y)] = c == '#'


def pr(ex):
    l = []
    for k, v in D.items():
        if v == ex:
            l.append(k)

    print_coords(l)

def f(x, y):
    r = ''.join('01'[D[(x+dx, y+dy)]] for dy in range(-1, 2) for dx in range(-1, 2))
    return alg[int(r, 2)] == '#'


for time in range(50):
    ND = defaultdict(lambda time=time: time % 2 == 0)

    xs, ys = zip(*D)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    N = 1
    for y in range(miny-N, maxy+N+1):
        for x in range(minx-N, maxx+N+1):
            ND[(x, y)] = f(x, y)
    D = ND

    if time in (1, 49):
        prints(sum(D.values()))
