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

best_row = max(max(count(Point.of(-1, y), 0), count(Point.of(W, y), 2)) for y in range(H))
best_col = max(max(count(Point.of(x, -1), 3), count(Point.of(x, H), 1)) for x in range(W))
print(f"Part 2: {max(best_row, best_col)}")
