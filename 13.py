#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


r = 0

dots = []

while True:
    s = input()
    if s == '': break
    dots.append(tuple(ints(s)))

try:
    while True:
        s = input()
        c = next(ints(s))
        is_x = 'x' in s

        ndots = []

        for a, b in dots:
            if is_x:
                if a >= c:
                    a = 2 * c - a
            else:
                if b >= c:
                    b = 2 * c - b

            ndots.append((a, b))

        dots = list(set(ndots))
except EOFError: pass


mx = max(x for x, y in dots)
my = max(y for x, y in dots)

s = set(dots)

for y in range(my+1):
    for x in range(mx+1):
        print('#' if (x, y) in dots else '.', end='')
    print()
