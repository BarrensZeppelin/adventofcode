#! /usr/bin/env pypy3
# >100/41
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

S = 'ABCD'
D = dict(zip(S, (1, 10, 100, 1000)))
dest = dict(zip(S, (3, 5, 7, 9)))

P = tuple([] for _ in range(4))

for y, l in enumerate(lines()):
    for x, c in enumerate(l):
        if c in S:
            P[S.index(c)].append((x, y))

def fix(state):
    return tuple(tuple(sorted(l)) for l in state)

def mkrev(state):
    D = {}
    for i, l in enumerate(state):
        for x, y in l:
            D[(x, y)] = S[i]
    return D

P = fix(P)
dist = {P: 0}
Q = [(0, P)]

def push(nd, nstate):
    #nstate = fix(nstate)
    pd = dist.get(nstate, -1)
    if pd == -1 or nd < pd:
        dist[nstate] = nd
        heappush(Q, (nd, nstate))


def is_correct(rev, x, y):
    c = rev.get((x, y))
    if c is None: return False
    return x == dest[c]

W = len('#...B.......#')
H = max(y for l in P for _, y in l) + 1
print(H)

while Q:
    d, state = heappop(Q)
    if d > dist[state]: continue
    #print(d)

    ok = True

    rev = mkrev(state)

    for idx, l in enumerate(state):

        for i, (x, y) in enumerate(l):
            if y > 1:
                corr = is_correct(rev, x, y)
                ok &= corr

                #if not corr or y == 2 and not is_correct(rev, x, y+1):
                if not corr or any(not is_correct(rev, x, ny) for ny in range(y+1, H)):
                    #if y == 3 and (x, y-1) in rev: continue
                    if y > 2 and any((x, ny) in rev for ny in range(2, y)):
                        continue

                    for sig in (-1, 1):
                        nx = x + sig
                        while (nx, 1) not in rev and 1 <= nx < W-1:
                            if nx % 2 and 3 <= nx <= 9:
                                nx += sig
                                continue

                            nstate = list(state)
                            new_l = list(nstate[idx])
                            new_l[i] = (nx, 1)
                            nstate[idx] = tuple(sorted(new_l))

                            move = (abs(x - nx) + y - 1) * D[S[idx]]
                            push(d + move, tuple(nstate))

                            nx += sig
            else:
                ok = False
                assert y == 1

                nx = dest[S[idx]]
                if (nx, 2) in rev: continue

                rny = 2
                will_move = True
                for ny in range(3, H):
                    if (nx, ny) in rev:
                        if not is_correct(rev, nx, ny):
                            will_move = False
                            break
                    else:
                        rny = ny

                if not will_move: continue

                #if (nx, 2) in rev or ((nx, 3) in rev and not is_correct(rev, nx, 3)):
                #    continue

                ny = rny
                xmove = abs(nx - x)

                way = sign(nx - x)
                while x != nx:
                    x += way
                    if (x, y) in rev:
                        will_move = False
                        break

                if not will_move:
                    continue

                nstate = list(state)
                new_l = list(nstate[idx])
                new_l[i] = (nx, ny)
                nstate[idx] = tuple(sorted(new_l))

                """
                print_coords(rev, '.')
                print()
                print_coords(mkrev(nstate), '.')
                print('-' * W)
                """

                move = (xmove + abs(ny - y)) * D[S[idx]]
                push(d + move, tuple(nstate))

    if ok:
        print_coords(rev, '.')
        prints(d)
        break
