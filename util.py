# pyright: reportSelfClsParameterName=none, reportGeneralTypeIssues=none

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict, deque
from functools import lru_cache, total_ordering
from heapq import heapify, heappop, heappush, heappushpop, heapreplace
from itertools import combinations
from itertools import combinations_with_replacement as combr
from itertools import cycle, permutations, product, repeat
from typing import (Any, Callable, Collection, Generic, Iterable, Iterator,
                    Mapping, Sequence, TypeVar)

sys.setrecursionlimit(1 << 30)

# E N W S
DIR = ((1, 0), (0, 1), (-1, 0), (0, -1))
HEXDIR = ((2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1))


def ints(inp: str | None = None) -> Iterator[int]:
    return map(int, re.findall(r"-?\d+", inp or sys.stdin.read()))


def floats(inp: str | None = None) -> Iterator[float]:
    return map(float, re.findall(r"-?\d+(?:\.\d*)?", inp or sys.stdin.read()))


def lines(inp: str | None = None) -> list[str]:
    return (inp or sys.stdin.read()).splitlines()


def prints(*args):
    """
    Function for printing the solution to a puzzle.
    Also copies the solution to the clipboard.
    """
    from subprocess import run

    ans = " ".join(map(str, args))
    print(ans)
    run(["xsel", "-bi"], input=ans, check=True, text=True)
    print("(Copied to clipboard)")


_U = TypeVar("_U")


def tile(L: Sequence[_U], S: int) -> list[Sequence[_U]]:
    assert len(L) % S == 0
    return [L[i : i + S] for i in range(0, len(L), S)]


def rotate(M: Iterable[Iterable[_U]], times=1) -> list[list[_U]]:
    "Rotate matrix ccw"
    for _ in range(times % 4):
        M = list(map(list, zip(*M)))[::-1]  # type: ignore
    return M  # type: ignore


