# pyright: reportSelfClsParameterName=none, reportGeneralTypeIssues=none

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Collection, Generator, Hashable, Iterable, Iterator, Mapping, Sequence
from functools import cache, cached_property, lru_cache, total_ordering
from heapq import heapify, heappop, heappush, heappushpop, heapreplace
from itertools import chain, combinations, cycle, groupby, permutations, product, repeat, starmap
from itertools import combinations_with_replacement as combr
from pathlib import Path
from typing import Final, Generic, Literal, TypeVar, cast, no_type_check, overload

try:
    import rich.traceback

    rich.traceback.install(indent_guides=False)
    del rich
except ImportError:
    pass

sys.setrecursionlimit(1 << 30)

# E N W S
DIR = DIR_NORTHPOS = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIR_NORTHNEG = ((1, 0), (0, -1), (-1, 0), (0, 1))
HEXDIR = ((2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1))
OCTDIR = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def ints(inp: str | None = None) -> list[int]:
    return list(map(int, re.findall(r"-?\d+", inp or sys.stdin.read())))


def floats(inp: str | None = None) -> list[float]:
    return list(map(float, re.findall(r"-?\d+(?:\.\d*)?", inp or sys.stdin.read())))


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


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def cut_interval(
    left: int, right: int, cut_left: int, cut_right: int
) -> tuple[tuple[int, int] | None, tuple[int, int], tuple[int, int] | None] | None:
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
        self.c = list(c) if isinstance(c, tuple) else c

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
        return f"({', '.join(map(str, s))})"

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
        return Point([a + b for a, b in zip(s, o, strict=False)])

    def __sub__(s, o: Iterable[T]) -> Point[T]:
        return Point([a - b for a, b in zip(s, o, strict=False)])

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
        return sum(a * b for a, b in zip(s, o, strict=False))

    __matmul__ = dot

    def cross(a, b: Point[T]) -> T:
        # assert len(a) == 2
        return a.x * b.y - a.y * b.x

    def cross2(s, a: Point[T], b: Point[T]) -> T:
        "Positive result ⇒  b is left of s -> a"
        return (a - s).cross(b - s)

    def cross_3d(a, b: Point[T]) -> Point[T]:
        assert len(a) == 3
        return Point.of(a.y * b.z - a.z * b.y, -a.x * b.z + a.z * b.x, a.x * b.y - a.y * b.x)

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

    def neigh(self, dirs: Iterable[Iterable[T]] = DIR) -> list[Point[T]]:
        return [self + d for d in dirs]


_N = TypeVar("_N", bound=Hashable)
_W = TypeVar("_W", int, float)
_TAR = TypeVar("_TAR")
_ADJ: TypeAlias = Mapping[_N, Iterable[_TAR]] | Callable[[_N], Iterable[_TAR]]


def make_adj(edges: Iterable[Iterable[_N]], both=False) -> defaultdict[_N, list[_N]]:
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        if both:
            adj[b].append(a)
    return adj


def make_wadj(edges: Iterable[tuple[_N, _N, _W]], both=False) -> defaultdict[_N, list[tuple[_N, _W]]]:
    adj = defaultdict(list)
    for a, b, w in edges:
        adj[a].append((b, w))
        if both:
            adj[b].append((a, w))
    return adj


def bfs(adj: _ADJ[_N, _N], *starts: _N) -> tuple[dict[_N, int], list[_N], dict[_N, _N]]:
    assert starts
    if not callable(adj):
        adj = adj.__getitem__

    D, Q, prev = dict.fromkeys(starts, 0), [*starts], {s: s for s in starts}
    for i in Q:
        d = D[i]
        for j in adj(i):
            if j not in D:
                D[j] = d + 1
                prev[j] = i
                Q.append(j)
    return D, Q, prev


def dijkstra(
    adj: _ADJ[_N, tuple[_N, _W]], *starts: _N, inf: _W = 1 << 60, heuristic: Callable[[_N], _W] | None = None
) -> tuple[defaultdict[_N, _W], dict[_N, list[_N]]]:
    assert starts
    if not callable(adj):
        adj = adj.__getitem__

    zero = inf * 0
    D = defaultdict(lambda: inf, dict.fromkeys(starts, zero))
    Q, prev = [(zero + (0 if heuristic is None else heuristic(s)), s) for s in D], {s: [s] for s in D}
    heapify(Q)
    V: set[_N] = set()
    while Q:
        _, i = heappop(Q)
        if i in V:
            continue
        V.add(i)
        d = D[i]
        for j, w in adj(i):
            nd = d + w
            if j not in V and nd < D[j]:
                D[j], prev[j] = nd, [i]
                heappush(Q, (nd + (0 if heuristic is None else heuristic(j)), j))
            elif nd == D[j]:
                prev[j].append(i)
    return D, prev


def conn_components(adj: Mapping[_N, Iterable[_N]]) -> list[list[_N]]:
    V: set[_N] = set()
    res: list[list[_N]] = []
    for i in list(adj):
        if i not in V:
            Q = [i]
            V.add(i)
            for i in Q:
                for j in adj[i]:
                    if j not in V:
                        V.add(j)
                        Q.append(j)
            res.append(Q)
    return res


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


class fungraph(Generic[_N]):
    def __init__(self, start: _N, next: Callable[[_N], _N], max_steps=10**6) -> None:
        self.start: Final = start
        self.next: Final = next
        self.max_steps: Final = max_steps
        self.dist: Final[dict[_N, int]] = {start: 0}
        self.nodes: Final[list[_N]] = [start]
        self.cycle: tuple[int, int] | None = None

    def __getitem__(self, k: int) -> _N:
        assert k >= 0
        if k < len(self.nodes):
            return self.nodes[k]
        if self.cycle is None:
            i = len(self.nodes) - 1
            n = self.nodes[i]
            for j in range(i + 1, k + 1):
                if j >= self.max_steps:
                    raise ValueError(f"Too many steps ({j})")

                n = self.next(n)
                if (pd := self.dist.get(n, -1)) != -1:
                    self.cycle = pd, j - pd
                    break
                self.dist[n] = j
                self.nodes.append(n)
            else:
                return n

        a, clen = self.cycle
        return self.nodes[a + (k - a) % clen]


class UF(Generic[_N]):
    def __init__(self):
        self.parent: Final[dict[_N, _N]] = {}

    def find(self, x: _N) -> _N:
        if (px := self.parent.setdefault(x, x)) != x:
            self.parent[x] = px = self.find(px)
        return px

    def join(self, x: _N, y: _N) -> bool:
        if (px := self.find(x)) == (py := self.find(y)):
            return False
        self.parent[px] = py
        return True


_U = TypeVar("_U")
_T = TypeVar("_T")
_MISSING: Final = object()


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
        M = list(map(list, zip(*M, strict=True)))[::-1]
    return M  # type: ignore


def print_coords(L: Collection[Iterable[int]], empty=" "):
    xs, ys = zip(*L, strict=True)
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
        while not f(hi + offset):
            lo, hi = hi, hi * 2
    else:
        hi -= offset

    assert lo <= hi
    while lo < hi:
        m = (lo + hi) // 2
        if f(m + offset):
            hi = m
        else:
            lo = m + 1

    return lo + offset


def binary_search_float(f: Callable[[float], bool], lo: float, hi: float | None = None, eps=1e-8) -> float:
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


class Grid(list[list[_U]]):
    def __init__(self, rows: Iterable[Iterable[_U]]) -> None:
        super().__init__([row if isinstance(row, list) else list(row) for row in rows])
        self.H: Final = len(self)
        self.W: Final = len(self[0])

    @classmethod
    def empty(cls, W: int, H: int, fill: _U) -> Grid[_U]:
        return cls([fill] * W for _ in range(H))

    @overload
    def inbounds(self, x: int, y: int) -> bool: ...
    @overload
    def inbounds(self, p: Iterable[int], /) -> bool: ...
    @no_type_check
    def inbounds(self, x, y=None):
        if y is None:
            x, y = x
        return 0 <= x < self.W and 0 <= y < self.H

    @overload
    def at(self, p: Iterable[int]) -> _U: ...
    @overload
    def at(self, p: Iterable[int], default: _T) -> _U | _T: ...
    def at(self, p: Iterable[int], default: _T = _MISSING) -> _U | _T:
        x, y = p
        if self.inbounds(x, y):
            return self[y][x]
        elif default is _MISSING:
            raise IndexError(f"{p} is out of bounds (W={self.W}, H={self.H})")
        else:
            return default

    __call__ = at

    def set(self, p: Iterable[int], v: _U) -> None:
        x, y = p
        self[y][x] = v

    @cached_property
    def rev(self) -> dict[_U, list[Point[int]]]:
        res: dict[_U, list[Point[int]]] = {}
        for y, row in enumerate(self):
            for x, c in enumerate(row):
                res.setdefault(c, []).append(Point([x, y]))
        return res

    def points(self) -> list[Point[int]]:
        return [Point([x, y]) for y in range(self.H) for x in range(self.W)]

    def adj(
        self, dirs: Iterable[tuple[int, int]] = DIR, pred: Callable[[Point[int], Point[int]], bool] | None = None
    ) -> dict[Point[int], list[Point[int]]]:
        return {
            p: [np for d in dirs if self.inbounds(np := p + d) and (pred is None or pred(p, np))]
            for p in self.points()
        }


_parser = argparse.ArgumentParser()
_parser.add_argument("--sample", "-s", action="store_true")
_parser.add_argument("--clear", "-c", "-f", action="store_true")
_ARGS, _EXTRA = _parser.parse_known_args()


def _get_day() -> int:
    match = re.fullmatch(r"(\d\d).*\.py", os.path.basename(sys.argv[0]))
    assert match is not None
    return int(match.group(1))


def replace_stdin():
    if not sys.stdin.isatty() or _EXTRA:
        return

    day = _get_day()
    p = Path(f"{day:02}.{'sin' if _ARGS.sample else 'in'}")
    if _ARGS.sample:
        if _ARGS.clear and p.exists():
            p.unlink()

        if not p.exists():
            import subprocess

            sample = subprocess.run(["xsel", "-b"], capture_output=True, text=True, check=True).stdout
            assert sample.strip(), "Clipboard is empty!"
            print(f"Writing sample\n-----\n{sample.strip()}\n-----\nto {p}")
            p.write_text(sample)

    sys.stdin = p.open()


def submit(answer: object, *, part: Literal[1, 2] | None = None, day: int | None = None):
    import contextlib
    import json

    from get_input import submit_answer

    STATE_FILE = Path(__file__).parent / ".state"

    if day is None:
        day = _get_day()

    assert 1 <= day <= 25, day

    state: dict[str, object] = {}
    if part is None:
        with contextlib.suppress(FileNotFoundError):
            state = json.loads(STATE_FILE.read_text())

        part = 2 if str(day) in state else 1

    assert sys.__stdin__ is not None

    answer = str(answer)
    print(f"Submitting day {day:02} part {part}:\n{answer}")
    if not sys.__stdin__.isatty():
        print("Aborting because stdin is redirected!")
        return
    if _ARGS.sample:
        print("Aborting because --sample is set!")
        return

    stdin_copy = sys.stdin
    try:
        sys.stdin = sys.__stdin__
        if (x := input("[yN] ").lower()) != "y":
            print(f"Aborting due to input: {x}")
            return
    finally:
        sys.stdin = stdin_copy

    response = submit_answer(day, part, answer)
    Path(".response").write_text(response)
    try:
        page = re.findall(r"article\>(.*)\</article", response, re.DOTALL)[0]
        page = re.sub(r"\<a href.*?\>(.*?)\</a\>", r"\1", page)
        page = page.replace("<p>", "").replace("</p>", "").replace("  ", "\n")
    except:
        print(response)
        raise
    else:
        print(page)

        if part == 1 and ("That's the right answer" in page or 'class="day-success"' in page):
            state[str(day)] = 0
            STATE_FILE.write_text(json.dumps(state))
            print("Updated state!")

        if match := re.search(r"rank \d+", page):
            print(match.group(0))
