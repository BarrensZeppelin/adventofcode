#!/usr/bin/env pypy3
# 266/68

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

G = lines()

W, H = len(G[0]), len(G)
for y, l in enumerate(G):
    for x, c in enumerate(l):
        if c == "S":
            start = (x, y)

# def adj(p):
#     for nx, ny in neighbours(*p):
#         if G[ny % H][nx % W] != "#":
#             yield (nx, ny)

@lru_cache(maxsize=None)
def bfs(mirr, *starts):
    assert starts

    D, Q, prev = {}, [*starts], {}
    for s in starts:
        D[s], prev[s] = 0, s

    for i in Q:
        d = D[i]
        for j in adj[i]:
            if j not in D:
                D[j] = d + 1
                prev[j] = i
                Q.append(j)
    return D, Q, prev

steps = 26501365
# SD, _, _ = bfs(0, start)

# res += sum(1 for d in SD.values() if d % 2 == 1)

assert start[0] + W // 2 + 1 == W

assert set(G[start[1]]) == {"S", "."}
assert set(G[y][start[0]] for y in range(H)) == {"S", "."}
lim = binary_search(lambda x: x * W > steps, 0)

for mirror in range(2):
    adj = grid_adj_2d(W, H, DIR, lambda _, p2: G[p2[1]][p2[0]] != "#")

    def get_startpos(px, py):
        if py == 0:
            nsy = start[1]
        else:
            mov = (H // 2) + 1 + (abs(py) - 1) * H
            nsy = start[1] + mov * sign(py)

        if px == 0:
            nsx = start[0]
        else:
            mov = (H // 2) + 1 + (abs(px) - 1) * H
            nsx = start[0] + mov * sign(px)
        return (nsx, nsy)


    for py in range(-lim, lim):
        if py % 1000 == 0: print(py, lim)
        def above(p):
            nsx, nsy = get_startpos(p, py)
            sd = abs(nsx - start[0]) + abs(nsy - start[1])
            return sd > steps

        px = binary_search(above, 0)
        hard = 0
        while px >= 0:
            if mirror == 1 and px == 0:
                break

            nsx, nsy = get_startpos(px, py)
            ns = (nsx, nsy)
            sd = abs(nsx - start[0]) + abs(nsy - start[1])
            if sd > steps:
                px -= 1
                continue
            rem = steps - sd
            # assert sd % 2 == 1, (sd, ns, px, py, rem)

            HD, _, _ = bfs(mirror, (nsx % W, nsy % H))
            if rem > 500 and px > 10:
                jump = (px - 1) // 2
                res += len(HD) * jump
                px -= jump * 2
                continue

            hard += 1
            res += sum(1 for d in HD.values() if d <= rem and d % 2 == rem % 2)
            # if rem >

            # print(rem)
            px -= 1
        # print(py, hard)

    G = [l[::-1] for l in G]


# print(start, D)
# print_coords({p: "O" if d <= 6 and d % 2 == 0 else "X" for p, d in D.items()})
# prints(sum(1 for d in D.values() if d <= 64 and d % 2 == 0))

prints(res)
