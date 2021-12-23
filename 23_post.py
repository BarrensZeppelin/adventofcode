#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

S = 'ABCD'
D = (1, 10, 100, 1000)
dest = dict(zip(S, (3, 5, 7, 9)))

Pnt = Tuple[int, int]
P: Tuple[List[Pnt], ...] = tuple([] for _ in range(4))

for y, line in enumerate(lines()):
    for x, c in enumerate(line):
        if c in S:
            P[S.index(c)].append((x, y))


def mkrev(state):
    "Build a reverse mapping from positions to amphipods"
    return {p: S[i] for i, l in enumerate(state) for p in l}

def new_state(state, idx, i, new_p):
    """
    Make a new normalised state from 'state' where the i'th amphipod of type
    'idx' is at 'new_p'
    """
    nstate = list(state)
    new_l = list(nstate[idx])
    new_l[i] = new_p
    nstate[idx] = tuple(sorted(new_l))
    return tuple(nstate)

P = tuple(tuple(sorted(l)) for l in P)
dist = {P: 0}
Q = [(0, P)]

def push(nd, nstate):
    pd = dist.get(nstate, -1)
    if pd == -1 or nd < pd:
        dist[nstate] = nd
        heappush(Q, (nd, nstate))


def is_correct(rev, x, y):
    c = rev.get((x, y))
    return c is not None and x == dest[c]

W = len('#...B.......#')
H = max(y for l in P for _, y in l) + 1

def printr(rev):
    S = [['.'] * W for _ in range(H)]
    for (x, y), c in rev.items():
        S[y][x] = c

    print('\n'.join(''.join(s) for s in S))

while Q:
    d, state = heappop(Q)
    if d > dist[state]: continue

    ok = True
    rev = mkrev(state)

    up_moves = []

    for idx, l in enumerate(state):
        for i, (x, y) in enumerate(l):
            if y > 1:
                corr = is_correct(rev, x, y)
                ok &= corr

                # We have to move up if any amphipod below us (including ourself)
                # is in the wrong position.
                if any(not is_correct(rev, x, ny) for ny in range(y, H)):
                    # We cannot move up if there is an amphipod above us
                    if any((x, ny) in rev for ny in range(2, y)):
                        continue

                    for sig in (-1, 1):
                        nx = x + sig
                        while (nx, 1) not in rev and 1 <= nx < W-1:
                            if nx % 2 and 3 <= nx <= 9:
                                nx += sig
                                continue

                            move = (abs(x - nx) + y - 1) * D[idx]
                            up_moves.append((d + move, new_state(state, idx, i, (nx, 1))))

                            nx += sig
            else:
                ok = False
                nx = dest[S[idx]]

                rny = 2
                will_move = True
                for ny in range(2, H):
                    if (nx, ny) in rev:
                        if not is_correct(rev, nx, ny):
                            will_move = False
                            break
                    else:
                        rny = ny

                if not will_move: continue

                xmove = abs(nx - x)
                way = sign(nx - x)
                while x != nx:
                    x += way
                    if (x, y) in rev:
                        will_move = False
                        break

                if not will_move: continue

                move = (xmove + abs(rny - y)) * D[idx]
                push(d + move, new_state(state, idx, i, (nx, rny)))
                # It is always optimal to move an amphipod into its final position
                break
        else:
            for nd, nstate in up_moves:
                push(nd, nstate)


    if ok:
        printr(rev)
        prints(d)
        break
