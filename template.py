#!/usr/bin/env pypy3
from __future__ import annotations

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

"""
L = sys.stdin.read().split("\n\n")

tile, rotate, Point, sign
"""

res = 0

for l in lines():
    pass

submit(1, res)
# submit(2, res)
# prints(res)
