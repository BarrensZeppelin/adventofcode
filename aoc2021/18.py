#! /usr/bin/env pypy3
# 23/19
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

L = lines()
res = eval(L[0])

def addl(a, v):
    assert isinstance(a, list)
    if isinstance(a[1], int):
        a[1] += v
    else:
        addl(a[1], v)

def addr(a, v):
    assert isinstance(a, list)
    if isinstance(a[0], int):
        a[0] += v
    else:
        addr(a[0], v)

def explode(a, depth=0):
    if isinstance(a, list):
        if depth == 4:
            return True, a[0], a[1]

        r = explode(a[0], depth+1)
        if r is not None:
            imm, b, c = r
            if imm:
                a[0] = 0

            if isinstance(a[1], int):
                a[1] += c
            else:
                addr(a[1], c)

            c = 0
            return False, b, c

        r = explode(a[1], depth+1)
        if r is not None:
            imm, b, c = r
            if imm:
                a[1] = 0

            if isinstance(a[0], int):
                a[0] += b
            else:
                addl(a[0], b)

            b = 0
            return False, b, c

def split(a):
    if isinstance(a, list):
        for i in range(2):
            if isinstance(a[i], int):
                if a[i] >= 10:
                    a[i] = [a[i] // 2, (a[i] + 1) // 2]
                    return True
            elif split(a[i]):
                return True


def reduce():
    while True:
        if explode(res):
            pass
        elif split(res):
            pass
        else:
            break

def magn(a):
    if isinstance(a, int):
        return a

    return 3 * magn(a[0]) + 2 * magn(a[1])

print(res)
reduce()
print(res)

for l in L[1:]:
    b = eval(l)
    res = [res, b]
    reduce()
    print(res)


print(magn(res))


best = 0
for i in range(len(L)):
    for j in range(len(L)):
        if i != j:
            res = [eval(L[i]), eval(L[j])]
            reduce()
            best = max(best, magn(res))

prints(best)
