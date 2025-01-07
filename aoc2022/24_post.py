#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


map = lines()
H = len(map)
W = len(map[0])

BP = dict(zip(">v<^ ", [Point.of(*d) for d in DIR] + [Point.of(0, 0)]))

start = Point.of(map[0].index("."), 0)
end = Point.of(map[-1].index("."), H - 1)

# There are no vertical blizzards in the start or end column
assert not set("^v") & {map[y][p.x] for y in range(H) for p in (start, end)}

period = math.lcm(H - 2, W - 2)
p11 = Point.of(1, 1)


def blizz(time, p):
    if p.y in (0, H - 1):
        return False
    for c, d in BP.items():
        np = p - d * time - p11
        if map[np.y % (H - 2) + 1][np.x % (W - 2) + 1] == c:
            return True

    return False

def bfs(st: int, start: Point[int], end: Point[int]):
    Q = [(st, start)]
    V = {(st % period, start)}
    for i, p in Q:
        if p == end:
            return i

        j = i + 1
        for d in BP.values():
            np = p + d

            if (
                0 <= np.y < H
                and map[np.y][np.x] != "#"
                and (j % period, np) not in V
                and not blizz(j, np)
            ):
                V.add((j % period, np))
                Q.append((j, np))
    else:
        assert False


p1 = bfs(0, start, end)
print(f"Part 1: {p1}")

p2 = bfs(bfs(p1, end, start), start, end)
print(f"Part 2: {p2}")
