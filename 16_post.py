#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

G = lines()
W, H = len(G[0]), len(G)

Node = tuple[Point, int]

def adj(n: Node) -> Iterable[Node]:
    p, d = n
    di = DIR_NORTHNEG[d]
    np = p + di
    if not 0 <= np.x < W or not 0 <= np.y < H:
        return ()
    match G[np.y][np.x]:
        case ".": return (np, d),
        case c if c in "/\\":
            dir = "/ \\".index(c)-1
            if d&1: dir *= -1
            return (np, (d-dir)%4),
        case c:
            if d&1 == "-|".index(c):
                return (np, d),
            return (np, (d-1)%4), (np, (d+1)%4)


def count(start: Point, sd: int) -> int:
    _, Q, _ = bfs(adj, (start, sd))
    return len({p for p, _ in Q})-1

print(f"Part 1: {count(Point.of(-1, 0), 0)}")

init = [
    *((Point.of(-1, y), 0) for y in range(H)),
    *((Point.of(W, y), 2) for y in range(H)),
    *((Point.of(x, -1), 3) for x in range(W)),
    *((Point.of(x, H), 1) for x in range(W)),
]

# print(f"Part 2: {max(starmap(count, init))}")

def find_SCC(*starts: Node):
    SCC, S, P = [], [], []
    depth = defaultdict[Node, int](int)
    stack = [(node, False) for node in starts]
    while stack:
        node, done = stack.pop()
        if done:
            d = depth[node] - 1
            if P[-1] > d:
                SCC.append(S[d:])
                del S[d:], P[-1]
                for node in SCC[-1]:
                    depth[node] = -1
        elif depth[node] > 0:
            while P[-1] > depth[node]:
                P.pop()
        elif depth[node] == 0:
            S.append(node)
            P.append(len(S))
            depth[node] = len(S)
            stack.append((node, True))
            for n in adj(node):
                stack.append((n, False))
    return SCC

def point_id(p: Point) -> int:
    return (p.x+1) + (p.y+1)*(W+2)

N = (W+2)*(H+2)
B = ceildiv(N, 63)
POPCNT = [0] * (1<<16)
for i in range(1<<16):
    POPCNT[i] = POPCNT[i>>1] + (i&1)
PMASK = (1<<16)-1

class BitSet:
    __slots__ = ("l",)

    def __init__(self):
        self.l = [0] * B

    def __setitem__(self, p: int, v: bool):
        b, p = divmod(p, 63)
        if v:
            self.l[b] |= 1 << p
        else:
            self.l[b] &= ~(1 << p)

    def __ior__(self, o: "BitSet") -> "BitSet":
        for b, x in enumerate(o.l):
            self.l[b] |= x
        return self

    def popcount(self) -> int:
        r = 0
        for x in self.l:
            while x:
                r += POPCNT[x&PMASK]
                x >>= 16
        return r


sccs = find_SCC(*init)
node_to_scc = {node: scc for scc, nodes in enumerate(sccs) for node in nodes}
DP = [BitSet() for _ in sccs]

for si, (scc, dp) in enumerate(zip(sccs, DP)):
    for node in scc:
        dp[point_id(node[0])] = True

        for node2 in adj(node):
            sj = node_to_scc[node2]
            if si != sj:
                assert sj < si, (si, sj)
                dp |= DP[sj]

best = max(DP[node_to_scc[n]].popcount() for n in init)-1
print(f"Part 2: {best}")
