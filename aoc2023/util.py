# pyright: reportSelfClsParameterName=none, reportGeneralTypeIssues=none

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict, deque
from functools import cache, lru_cache, total_ordering
from heapq import heapify, heappop, heappush, heappushpop, heapreplace
from itertools import combinations
from itertools import combinations_with_replacement as combr
from itertools import cycle, groupby, permutations, product, repeat, starmap
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from typing import Generic, Hashable, TypeVar

sys.setrecursionlimit(1 << 30)

# E N W S
DIR = DIR_NORTHPOS = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIR_NORTHNEG = ((1, 0), (0, -1), (-1, 0), (0, 1))
HEXDIR = ((2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1))
OCTDIR = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def ints(inp: str | None = None) -> Iterator[int]:
    return map(int, re.findall(r"-?\d+", inp or sys.stdin.read()))


def floats(inp: str | None = None) -> Iterator[float]:
    return map(float, re.findall(r"-?\d+(?:\.\d*)?", inp or sys.stdin.read()))


def lines(inp: str | None = None) -> list[str]:
    return (inp or sys.stdin.read()).splitlines()


def prints(*args, copy=len(sys.argv) == 1):
    """
    Function for printing the solution to a puzzle.
    Also copies the solution to the clipboard.
    """
    from subprocess import run

    ans = " ".join(map(str, args))
    print(ans)
    if copy:
        run(["xsel", "-bi"], input=ans, check=True, text=True)
        print("(Copied to clipboard)")


def ceildiv(a: int, b: int) -> int: return -(a // -b)


def cut_interval(left: int, right: int, cut_left: int, cut_right: int) -> None | tuple[
    tuple[int, int] | None,
    tuple[int, int],
    tuple[int, int] | None,
]:
    """
    Cuts an [incl, excl) interval with another.
    Returns None if there is no overlap, otherwise returns the three intervals:
    1. The part of the interval to the left of the cut (may be None)
    2. The intersection of the two intervals
    3. The part of the interval to the right of the cut (may be None)
    """
    assert left < right
    if right <= cut_left or cut_right <= left:
        return None
    return (
        (left, cut_left) if left < cut_left else None,
        (max(left, cut_left), min(right, cut_right)),
        (cut_right, right) if cut_right < right else None,
    )


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    "Sorts the intervals and merges overlapping ones"
    S: list[tuple[int, int]] = []
    for a, b in sorted(intervals):
        if S and S[-1][1] >= a:
            a, c = S.pop()
            b = max(b, c)
        S.append((a, b))
    return S


T = TypeVar("T", int, float)


def sign(x: T) -> int:
    return (x > 0) - (x < 0)


@total_ordering
class Point(Generic[T]):
    c: list[T]
    __slots__ = ("c",)

    def __init__(self, c: list[T] | tuple[T, ...]):
        if isinstance(c, tuple): c = list(c)
        self.c = c

    @classmethod
    def of(cls, *c: T) -> Point[T]:
        return cls(list(c))

    # Points are generally immutable except that you can set coordinates

    @property
    def x(s) -> T:
        return s.c[0]

    @x.setter
    def x(s, v: T):
        s.c[0] = v

    @property
    def y(s) -> T:
        return s.c[1]

    @y.setter
    def y(s, v: T):
        s.c[1] = v

    @property
    def z(s) -> T:
        return s.c[2]

    @z.setter
    def z(s, v: T):
        s.c[2] = v

    # Standard object methods

    def __lt__(s, o: Point[T]) -> bool:
        return s.c < o.c

    def __eq__(s, o) -> bool:
        return isinstance(o, Point) and s.c == o.c

    def __hash__(s) -> int:
        return hash(tuple(s.c))

    def __str__(s) -> str:
        return f'({", ".join(map(str, s))})'

    def __repr__(s) -> str:
        return f"Point({s.c})"

    def __len__(s) -> int:
        return len(s.c)

    def __iter__(s) -> Iterator[T]:
        return iter(s.c)

    def __getitem__(s, key):
        return s.c[key]

    def map(s, f: Callable[[T], T]) -> Point[T]:
        return Point([*map(f, s)])

    # Geometry stuff

    def __add__(s, o: Iterable[T]) -> Point[T]:
        return Point([a + b for a, b in zip(s, o)])

    def __sub__(s, o: Iterable[T]) -> Point[T]:
        return Point([a - b for a, b in zip(s, o)])

    def __neg__(s) -> Point[T]:
        return Point([-x for x in s])

    def __abs__(s) -> Point[T]:
        return s.map(abs)

    def __mul__(s, d: T) -> Point[T]:
        return Point([a * d for a in s])

    __rmul__ = __mul__

    def __floordiv__(s, d: T) -> Point[T]:
        return Point([a // d for a in s])

    def __truediv__(s, d: T) -> Point[float]:
        return Point([a / d for a in s])

    def dot(s, o: Iterable[T]) -> T:
        return sum(a * b for a, b in zip(s, o))

    __matmul__ = dot

    def cross(a, b: Point[T]) -> T:
        # assert len(a) == 2
        return a.x * b.y - a.y * b.x

    def cross2(s, a: Point[T], b: Point[T]) -> T:
        "Positive result ⇒  b is left of s -> a"
        return (a - s).cross(b - s)

    def cross_3d(a, b: Point[T]) -> Point[T]:
        assert len(a) == 3
        return Point.of(
            a.y * b.z - a.z * b.y, -a.x * b.z + a.z * b.x, a.x * b.y - a.y * b.x
        )

    def cross2_3d(s, a: Point[T], b: Point[T]) -> Point[T]:
        return (a - s).cross_3d(b - s)

    def manh_dist(s) -> T:
        return sum(s.map(abs))

    def dist2(s) -> T:
        return sum(x * x for x in s)

    def dist(s) -> float:
        return s.dist2() ** 0.5

    def angle(s) -> float:
        assert len(s) == 2
        return math.atan2(s.y, s.x)

    def perp(s) -> Point[T]:
        "Rotate ccw 90°"
        assert len(s) == 2
        return Point([-s.y, s.x])

    def rotate(s, a: float) -> Point[float]:
        assert len(s) == 2
        co, si = math.cos(a), math.sin(a)
        return Point([s.x * co - s.y * si, s.x * si + s.y * co])


_N = TypeVar("_N", bound=Hashable)
_W = TypeVar("_W", int, float)


def make_adj(edges: Iterable[Iterable[_N]], both=False) -> defaultdict[_N, list[_N]]:
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        if both:
            adj[b].append(a)
    return adj


def make_wadj(
    edges: Iterable[tuple[_N, _N, _W]], both=False
) -> defaultdict[_N, list[tuple[_N, _W]]]:
    adj = defaultdict(list)
    for a, b, w in edges:
        adj[a].append((b, w))
        if both:
            adj[b].append((a, w))
    return adj


def bfs(
    adj: Mapping[_N, Iterable[_N]] | Callable[[_N], Iterable[_N]],
    *starts: _N
) -> tuple[dict[_N, int], list[_N], dict[_N, _N]]:
    assert starts
    if not callable(adj):
        adj = adj.__getitem__

    D, Q, prev = {}, [*starts], {}
    for s in starts:
        D[s], prev[s] = 0, s

    for i in Q:
        d = D[i]
        for j in adj(i):
            if j not in D:
                D[j] = d + 1
                prev[j] = i
                Q.append(j)
    return D, Q, prev


def dijkstra(
    adj, *starts: _N, inf: _W = 1 << 60,
    heuristic = None,
) -> tuple[defaultdict[_N, _W], dict[_N, _N]]:
    assert starts
    zero = inf * 0
    D: defaultdict[_N, _W] = defaultdict(lambda: inf)
    V = set()
    Q = []
    prev = {}
    for s in starts:
        D[s] = 0
        Q.append((zero + (0 if heuristic is None else heuristic(s)), s))
        prev[s] = s
    heapify(Q)
    while Q:
        _, i = heappop(Q)
        if i in V:
            continue
        V.add(i)
        d = D[i]
        for j, w in adj(i):
            nd = d + w
            if j not in V and nd < D[j]:
                D[j] = nd
                prev[j] = i
                heappush(Q, (nd + (0 if heuristic is None else heuristic(j)), j))
    return D, prev


def dijkstra_old(
    adj: Mapping[_N, Iterable[tuple[_N, _W]]], *starts: _N, inf: _W = 1 << 60
) -> tuple[defaultdict[_N, _W], dict[_N, _N]]:
    assert starts
    zero = inf * 0
    D: defaultdict[_N, _W] = defaultdict(lambda: inf)
    V = set()
    Q = []
    prev = {}
    for s in starts:
        D[s] = 0
        Q.append((zero, s))
        prev[s] = s
    heapify(Q)
    while Q:
        d, i = heappop(Q)
        if i in V:
            continue
        V.add(i)
        for j, w in adj[i]:
            nd = d + w
            if j not in V and nd < D[j]:
                D[j] = nd
                prev[j] = i
                heappush(Q, (nd, j))
    return D, prev


def make_path(frm: _N, to: _N, prev: Mapping[_N, _N]) -> list[_N]:
    assert to in prev
    path = [to]
    while path[-1] != frm:
        path.append(prev[path[-1]])
    return path[::-1]


def topsort(adj: Mapping[_N, Iterable[_N]]) -> tuple[list[_N], bool]:
    "Flag is true iff. graph is cyclic"
    indeg: defaultdict[_N, int] = defaultdict(int)
    for i, l in adj.items():
        indeg[i] += 0  # make sure all nodes are in indeg
        for j in l:
            indeg[j] += 1
    Q = [i for i in adj if indeg[i] == 0]
    for i in Q:
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                Q.append(j)
    return Q, len(Q) != len(indeg)


_U = TypeVar("_U")


def tile(L: Sequence[_U], S: int) -> list[Sequence[_U]]:
    assert len(L) % S == 0
    return [L[i : i + S] for i in range(0, len(L), S)]


def windows(L: Sequence[_U], S: int) -> list[Sequence[_U]]:
    assert len(L) >= S
    return [L[i : i + S] for i in range(len(L) - S + 1)]


def run_length_encoding(L: Iterable[_U]) -> list[tuple[_U, int]]:
    return [(c, len(list(g))) for c, g in groupby(L)]


def rotate(M: Iterable[Iterable[_U]], times=1) -> list[list[_U]]:
    "Rotate matrix ccw"
    for _ in range(times % 4):
        M = list(map(list, zip(*M)))[::-1]
    return M  # type: ignore


def print_coords(L: Collection[tuple[int, int]], empty=" "):
    xs, ys = zip(*L)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    print("X", min_x, max_x)
    print("Y", min_y, max_y)

    R = [[empty] * (max_x - min_x + 1) for _ in range(max_y - min_y + 1)]

    if isinstance(L, Mapping):
        for (x, y), c in L.items():
            assert len(c) == 1, ((x, y), c)
            R[y - min_y][x - min_x] = c
    else:
        for x, y in L:
            R[y - min_y][x - min_x] = "#"

    print(*map("".join, R), sep="\n")


def binary_search(f: Callable[[int], bool], lo: int, hi: int | None = None) -> int:
    "Returns the first i >= lo such that f(i) == True"
    lo, offset = 0, lo
    if hi is None:
        hi = 1
        while not f(hi+offset):
            lo, hi = hi, hi * 2
    else:
        hi -= offset

    assert lo <= hi
    while lo < hi:
        m = (lo + hi) // 2
        if f(m+offset):
            hi = m
        else:
            lo = m + 1

    return lo+offset


def binary_search_float(
    f: Callable[[float], bool], lo: float, hi: float | None = None, eps=1e-8
) -> float:
    if hi is None:
        assert lo >= 0.0
        hi = lo + 1
        while not f(hi):
            lo, hi = hi, hi * 2

    assert lo <= hi
    while hi - lo > eps:
        m = (lo + hi) / 2
        if f(m):
            hi = m
        else:
            lo = m

    return lo


def grid_adj_2d(
    W: int,
    H: int,
    dirs: Iterable[tuple[int, int]] = DIR,
    pred: Callable[[tuple[int, int], tuple[int, int]], bool] | None = None,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    adj = dict()
    for y in range(H):
        for x in range(W):
            l = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < W
                    and 0 <= ny < H
                    and (pred is None or pred((x, y), (nx, ny)))
                ):
                    l.append((nx, ny))
            adj[(x, y)] = l

    return adj


def neighbours(
    x: int, y: int, dirs: Iterable[tuple[int, int]] = DIR, V=None
) -> Iterator[tuple[int, int]]:
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if V is None or (nx, ny) in V:
            yield nx, ny
