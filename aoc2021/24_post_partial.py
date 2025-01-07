#! /usr/bin/env pypy3
from util import *

FS = sys.stdin.read()
secs = FS.split("inp w\n")[1:]
secs = [[s.split() for s in lines(sec)] for sec in secs]
N = len(secs)
assert N == 14


def add(x, y, vs):
    vs[x] += y(vs)

def mul(x, y, vs):
    vs[x] *= y(vs)

def div(x, y, vs):
    vs[x] //= y(vs)

def mod(x, y, vs):
    vs[x] %= y(vs)

def eql(x, y, vs):
    vs[x] = int(vs[x] == y(vs))


from functools import partial

fun_map = {fun.__name__: fun for fun in (add, mul, div, mod, eql)}

opt_secs = []
for sec in secs:
    funs = []
    for instr, x, y in sec:
        x = ord(x) - ord("w")
        yf = (
            (lambda vs, y=int(y): y)
            if y[0] == "-" or y.isdigit()
            else (lambda vs, i=(ord(y) - ord("w")): vs[i])
        )
        funs.append(partial(fun_map[instr], x, yf))

    opt_secs.append(funs)


cache = [set() for _ in range(14)]
VARS = "wxyz"


def run(sec_no: int, pz: int) -> int:
    if sec_no == N or pz > 10 ** 7:
        return 0 if pz == 0 else -1

    if pz in cache[sec_no]: return -1

    for w in range(9, 0, -1):
        # for w in range(1, 10):
        vs = [0] * 4
        vs[0] = w
        vs[3] = pz

        for fun in opt_secs[sec_no]: fun(vs)

        r = run(sec_no + 1, vs[-1])
        if r != -1:
            return r + w * 10 ** (N - sec_no - 1)

    cache[sec_no].add(pz)
    return -1


print(run(0, 0))
