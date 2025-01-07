#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

FS = sys.stdin.read()
secs = FS.split('inp w\n')[1:]
secs = [[s.split() for s in lines(sec)] for sec in secs]
N = len(secs)
assert N == 14

cache = [set() for _ in range(14)]
VARS = 'wxyz'

def run(sec_no: int, pz: int) -> int:
    if sec_no == N or pz > 10 ** 7:
        return 0 if pz == 0 else -1

    if pz in cache[sec_no]: return -1

    #for w in range(9, 0, -1):
    for w in range(1, 10):
        vs = [0] * 4
        vs[0] = vs[1] = w
        vs[3] = pz

        def get(x):
            if x[0] == '-' or x.isdigit():
                return int(x)
            return vs[ord(x) - ord('w')]

        for instr, *args in secs[sec_no]:
            if instr == 'inp':
                assert False
            else:
                x, y = args
                x = ord(x) - ord('w')
                y = get(y)

                if instr == 'add':
                    vs[x] += y
                elif instr == 'mul':
                    vs[x] *= y
                elif instr == 'div':
                    assert y != 0
                    vs[x] //= y
                elif instr == 'mod':
                    assert vs[x] >= 0
                    assert y > 0
                    vs[x] %= y
                elif instr == 'eql':
                    vs[x] = int(vs[x] == y)

        r = run(sec_no + 1, vs[-1])
        if r != -1:
            return r + w * 10 ** (N - sec_no - 1)

    cache[sec_no].add(pz)
    return -1

print(run(0, 0))
