#!/usr/bin/env pypy3
# 358/109
from __future__ import annotations

from util import *

replace_stdin()

res = 0

S = sys.stdin.read()
en = True
for i, c in enumerate(S):
    if S[i:].startswith("don't()"):
        en = False
    elif S[i:].startswith("do()"):
        en = True
    elif en and (match := re.match(r"mul\((\d+),(\d+)\)", S[i:])):
        a, b = map(int, match.groups())
        res += a * b

submit(res)
