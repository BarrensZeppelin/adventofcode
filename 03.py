#!/usr/bin/env pypy3
# 9/61

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

L = lines()

NUMS = {}
IDX = 0

for y, l in enumerate(L):
    W = len(l)
    i = 0
    num = 0
    def fix():
        global res, IDX
        IDX += 1
        j = i-1
        added = False
        while j >= 0:
            if not l[j].isdigit():
                break
            NUMS[(j, y)] = (num, IDX)
            if not added:
                for dx, dy in OCTDIR:
                    x, ny = j + dx, y + dy
                    if 0 <= x < W and 0 <= ny < len(L):
                        c = L[ny][x]
                        if not c.isdigit() and c != ".":
                            res += num
                            added = True
            j -= 1
    while i < W:
        if l[i].isdigit():
            num = num * 10 + int(l[i])
        else:
            if num:
                fix()
            num = 0
        i += 1
    if num:
        fix()

res = 0
print(NUMS)

for y, l in enumerate(L):
    for x, c in enumerate(l):
        if c == "*":
            nums = set()
            for dx, dy in OCTDIR:
                nx, ny = x + dx, y + dy
                v = NUMS.get((nx, ny))
                if v is not None:
                    nums.add(v)

            if len(nums) == 2:
                a, _ = nums.pop()
                b, _ = nums.pop()
                res += a * b

prints(res)
