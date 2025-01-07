#!/usr/bin/env pypy3
# 11/1

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

ws, psb = sys.stdin.read().split("\n\n")

workflows = {}
for line in lines(ws):
    name, rest = line.split("{")
    rest = rest[:-1]
    ps = rest.split(",")
    workflows[name] = [p.split(":") for p in ps]
    # print(name, workflows[name])

res = 0

def process(wn, part):
    if wn == "A":
        return True
    elif wn == "R":
        return False
    rule = workflows[wn]
    for r in rule:
        if len(r) == 1:
            return process(r[0], part)

        if eval(r[0], locals=part):
            return process(r[1], part)
    assert False

for part in lines(psb):
    part = part[1:-1].split(",")
    d = {}
    for p in part:
        k, v = p.split("=")
        d[k] = int(v)

    if process("in", d):
        res += sum(d.values())
    # print(d)

def process(wn, part):
    if wn == "A":
        return math.prod(b - a + 1 for a, b in part.values()) if all(a <= b for a, b in part.values()) else 0
        return True
    elif wn == "R":
        return 0
    rule = workflows[wn]
    curr = 0
    for r in rule:
        if len(r) == 1:
            curr += process(r[0], part)
            continue

        cond, res = r
        op = ">" if ">" in cond else "<"
        a, b = cond.split(op)
        lo, hi = part[a]
        p2 = dict(part)
        if op == ">":
            p2[a] = (max(lo, int(b) + 1), hi)
            curr += process(res, p2)
            part[a] = (lo, min(hi, int(b)))
        else:
            p2[a] = (lo, min(hi, int(b) - 1))
            curr += process(res, p2)
            part[a] = (max(lo, int(b)), hi)

        # if eval(r[0], locals=part):
            # return process(r[1], part)
    return curr
    assert False

prints(process("in", {k: (1, 4000) for k in "xmas"}))
