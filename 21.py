#!/usr/bin/env pypy3
# 930/332
from __future__ import annotations

from util import *

replace_stdin()

G = Grid(
    lines("""\
789
456
123
 0A""")
)

G2 = Grid(
    lines("""\
 ^A
<v>""")
)

res = 0


@cache
def g(d: Point[int], i: int):
    if i == 2:
        r = []
        if d.x:
            r.append((">" if d.x > 0 else "<") * abs(d.x))
        if d.y:
            r.append(("v" if d.y > 0 else "^") * abs(d.y))
        return "".join(r)
    r = []
    p = G2.crev["A"][0]
    if d.x:
        nc = "<" if d.x < 0 else ">"
        np = G2.crev[nc][0]
        r.extend(g(np - p, i + 1) + "A" * abs(d.x))
        # r += g(np - p, i + 1)
        p = np
    if d.y:
        nc = "^" if d.y < 0 else "v"
        np = G2.crev[nc][0]
        r.extend(g(np - p, i + 1) + "A" * abs(d.y))
        # r += g(np - p, i + 1)# * abs(d.y)
        p = np
    d = G2.crev["A"][0] - p
    # r += g(d, i + 1)
    r.extend(g(d, i + 1))
    r.append("A")
    # r.append("A")
    return "".join(r)
    # return r + 1
    # if i == 2:
    #     r = []
    #     if d.x:
    #         r.append((">" if d.x > 0 else "<") * abs(d.x))
    #     if d.y:
    #         r.append(("v" if d.y > 0 else "^") * abs(d.y))
    #     return "".join(r)
    # r = []
    # p = G2.crev["A"][0]
    # if d.x:
    #     nc = "<" if d.x < 0 else ">"
    #     np = G2.crev[nc][0]
    #     r.extend(g(np - p, i + 1) + "A" * abs(d.x))
    #     # r += g(np - p, i + 1)
    #     p = np
    # if d.y:
    #     nc = "^" if d.y < 0 else "v"
    #     np = G2.crev[nc][0]
    #     r.extend(g(np - p, i + 1) + "A" * abs(d.y))
    #     # r += g(np - p, i + 1)# * abs(d.y)
    #     p = np
    # d = G2.crev["A"][0] - p
    # # r += g(d, i + 1)
    # r.extend(g(d, i + 1))
    # r.append("A")
    # # r.append("A")
    # return "".join(r)
    # # return r + 1


@cache
def f(c: str, i: int):
    if i == 2:
        return c

    r = []
    p = G2.crev["A"][0]
    np = G2.crev[c][0]
    d = np - p
    if d.x:
        x = f(">" if d.x > 0 else "<", i + 1)  # + f("A", i+1)
        r.extend(x * abs(d.x))
    if d.y:
        x = f("v" if d.y > 0 else "^", i + 1)  # + f("A", i+1)
        r.extend(x * abs(d.y))
    r.append(f("A", i + 1))
    d = -d
    print(c, i, r)
    return "".join(r)


@cache
def fuck(s: str, i: int):
    # print(len(s))
    if i == 25:
        return len(s)

    # print(s, i)
    p = G2.crev["A"][0]
    moves = 0
    # fr = ""
    for c in s:
        np = G2.crev[c][0]
        d = np - p
        r = []
        xs = ys = ""
        if d.x:
            xs = (">" if d.x > 0 else "<") * abs(d.x)
        if d.y:
            ys = ("v" if d.y > 0 else "^") * abs(d.y)

        opts = []
        if G2(p + (d.x, 0)) != " ":
            opts.append(fuck(xs + ys + "A", i+1))
        if G2(p + (0, d.y)) != " ":
            opts.append(fuck(ys + xs + "A", i+1))
        # to_press: list[str] = []
        # if d.y:
        #     to_press.extend(("v" if d.y > 0 else "^") * abs(d.y))
        # if d.x:
        #     to_press.extend((">" if d.x > 0 else "<") * abs(d.x))
        # to_press.extend("A")
        r = min(opts)
        # ps, r = fuck("".join(to_press), i + 1)
        # m = g(d, 0)
        # r = "".join(r)
        moves += r
        p = np
    assert p == G2.crev["A"][0]
    return moves


for l in lines():
    ps = [G2.crev["A"][0]] * 2
    p = G.crev["A"][0]
    moves = 0
    fr = ""
    for c in l:
        np = G.crev[c][0]
        d = np - p
        r = []
        to_press: list[str] = []
        dd = 0
        xs = ys = ""
        if d.x:
            xs = (">" if d.x > 0 else "<") * abs(d.x)
        if d.y:
            ys = ("v" if d.y > 0 else "^") * abs(d.y)

        opts = []
        if G(p + (d.x, 0)) != " ":
            opts.append(fuck(xs + ys + "A", dd))
        if G(p + (0, d.y)) != " ":
            opts.append(fuck(ys + xs + "A", dd))

        r = min(opts)
        # r = fuck("".join(to_press), dd)
        # m = g(d, 0)
        # r = "".join(r)
        # fr += r
        # print(p, np, d, r)
        moves += r
        p = np
        # moves += g(d, 0)

    print(l, moves, fr)
    # assert len(fr) == moves
    res += moves * int(l[:-1])
    # break


submit(res)
