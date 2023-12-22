#!/usr/bin/env pypy3
# 74/71

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

P = Point

res = 0

bricks = []

for l in lines():
    a, b = l.split("~")
    a, b = map(lambda s: P.of(*map(int, s.split(","))), [a, b])
    for x, y in zip(a, b):
        assert x <= y
    bricks.append((a, b))

def key(b):
    a, b = b
    return a.z

bricks.sort(key=key)

X = defaultdict(dict)

# supports = Counter()
ban = set()

order = sorted(range(len(bricks)), key=lambda i: bricks[i][0].z)

adj = defaultdict(set)
belows = dict()

for i in order:
    s = chr(ord("A") + i)
    br = bricks[i]
    a, b = br
    z = a.z-1
    while z >= 1:
        below = set()
        for x in range(a.x, b.x+1):
            for y in range(a.y, b.y+1):
                if (x, y) in X[z]:
                    below.add(X[z][(x, y)])
            # if (x, y) in Y[z]:
            #     below.add(Y[z][(x, y)])

        if not below:
            z -= 1
        else:
            # print(s, "by", below)
            belows[i] = below
            for j in below:
                adj[j].add(i)
            # if len(below) == 1:
            #     for j in below:
            #         adj[j].add(i)
                # ban |= below
            # for i in below:
            #     supports[i] += 1
            break
    else:
        belows[i] = set()

    z += 1
    # print(s, a, b, z)

    for x in range(a.x, b.x+1):
        for y in range(a.y, b.y+1):
            for nz in range(z, z+b.z-a.z+1):
                X[nz][(x, y)] = i

cp = {i: set(b) for i, b in belows.items()}

for i in range(len(bricks)):
    Q = [i]
    for x in Q:
        for j in adj[x]:
            belows[j].remove(x)
            if not belows[j]:
                Q.append(j)

    res += len(Q)-1
    belows = {i: set(b) for i, b in cp.items()}
    # bfs(adj, i)[0])
    # print(i, adj[i])


prints(res)
exit()


print(ban)
prints(len(bricks) - len(ban))
