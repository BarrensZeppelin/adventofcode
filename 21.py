#! /usr/bin/env pypy3
# 108/18
import z3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


res = 0

adj = defaultdict(list)

do = {}

for l in lines():
    name, op = l.split(': ')
    ops = op.split()
    if len(ops) == 1:
        do[name] = int(op)
    else:
        adj[ops[0]].append(name)
        adj[ops[2]].append(name)
        do[name] = ops


L, cyclic = topsort(adj)
assert not cyclic

z3var = z3.Int("x")
s = z3.Solver()

do["humn"] = z3var

val = {}
for name in L:
    op = do[name]
    if not isinstance(op, list):
        val[name] = op
    else:
        a = val[op[0]]
        b = val[op[2]]

        if name == "root":
            s.add(a == b)

            assert s.check() == z3.sat
            prints(s.model().eval(z3var))
            exit()

        c = op[1]

        if c == "+":
            r = a + b
        elif c == "-":
            r = a - b
        elif c == "*":
            r = a * b
        else:
            r = a / b

        val[name] = r

prints(val["root"])
