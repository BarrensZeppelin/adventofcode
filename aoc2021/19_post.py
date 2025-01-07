#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

S = sys.stdin.read().split("\n\n")

scan = []
for i, s in enumerate(S):
    beacons = {tuple(ints(l)) for l in lines(s)[1:]}
    fp = sorted((Point.of(*a) - b).dist2() for a, b in combinations(beacons, r=2))
    scan.append((beacons, fp))

here = [scan.pop(0)]
P = [Point.of(0, 0, 0)]

F = [Point(l) for x in (-1, 1) for l in ([x, 0, 0], [0, x, 0], [0, 0, x])]
rotation_matrices = []
for a, b in permutations(F, r=2):
    if (a + b).manh_dist() != 2: continue
    c = a.cross_3d(b)

    # Make matrix with basis as columns (transposed basis rows)
    rotation_matrices.append(tuple(zip(a, b, c)))


def matvec(A: Sequence[Sequence[int]], v: Point[int]):
    return [sum(A[i][j] * v[j] for j in range(3)) for i in range(3)]


def mods(scanner: Sequence[Point[int]]) -> Iterable[List[List[int]]]:
    for m in rotation_matrices:
        yield [matvec(m, v) for v in scanner]


def fp_matches(a, b):
    cnt = 0
    i = 0
    for x in a:
        while i < len(b) and b[i] < x: i += 1
        if i < len(b) and b[i] == x:
            cnt += 1
            i += 1

    return cnt >= 12 * (12-1) // 2


def matches(next, prev):
    prev_points, prev_fp = prev
    if not fp_matches(next[1], prev_fp): return

    for me in mods(next[0]):
        for ref1 in prev_points:
            for ref2 in me:
                s2 = Point.of(*ref1) - ref2

                overlap = 0
                for i, ps in enumerate(me):
                    if overlap + len(me) - i < 12:
                        break

                    if tuple(s2 + ps) in prev_points:
                        overlap += 1

                        if overlap >= 12:
                            return me, s2


for prev in here:
    i = 0
    while i < len(scan):
        sc = scan[i]
        r = matches(sc, prev)
        if r is not None:
            scan.pop(i)
            me, s2 = r
            here.append(({tuple(s2 + p) for p in me}, sc[1]))
            P.append(s2)
        else:
            i += 1

assert not scan

print(len(set().union(*(points for points, _ in here))))

print(max((a - b).manh_dist() for a, b in combinations(P, r=2)))
