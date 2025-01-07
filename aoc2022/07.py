#! /usr/bin/env pypy3
# 76/85
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

def Node(parent=None):
    return [{}, parent]

root = Node()
_root = root

res = 0

L = lines()
i = 0
while i < len(L):
    l = L[i]
    assert l.startswith("$")
    p = l[1:].split()
    if len(p) == 2 and p[0] == "cd":
        d = p[1]
        if d == "..":
            root = root[1]
        else:
            if d not in root[0]:
                root[0][d] = Node(root)

            root = root[0][d]
        i += 1
    elif len(p) == 1 and p[0] == "ls":
        i += 1

        while i < len(L) and not L[i].startswith("$"):
            a, d = L[i].split()
            if a == "dir":
                if d not in root[0]:
                    root[0][d] = Node(root)
            else:
                root[0][d] = int(a)

            i += 1

    else:
        assert False


LIM = 100000
AVA =  70000000
NEED = 30000000
def f(n):
    global res
    assert isinstance(n, list)

    cur = 0
    for d in n[0].values():
        if isinstance(d, list):
            cur += f(d)
        else:
            cur += d

    if cur <= LIM:
        res += cur

    return cur

used = f(_root)
#print(used)

res = 1<<60

def g(n):
    global res
    assert isinstance(n, list)

    cur = 0
    for d in n[0].values():
        if isinstance(d, list):
            cur += g(d)
        else:
            cur += d

    if AVA - used + cur >= NEED:
        res = min(res, cur)

    return cur

g(_root)

prints(res)
