#! /usr/bin/env pypy3
# 92/90
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


res = 0



#DIR = ((1, 0), (0, -1), (-1, 0), (0, 1))
DIR = ((0, -1), (0, 1), (-1, 0), (1, 0))
elves = set()

for y, l in enumerate(lines()):
    for x, c in enumerate(l):
        if c == "#":
            elves.add(Point.of(x, y))


for round in range(1<<30):
    prop = Counter()
    mov = {}
    #print(DIR[(round)%4])
    for elf in elves:
        if all((elf + Point.of(*d)) not in elves for d in OCTDIR):
            continue

        for i in range(4):
            np = elf + DIR[(round+i)%4]

            #print(elf, np)
            ok = True
            if np.y == elf.y:
                for dy in range(-1, 2):
                    ok &= (np + Point.of(0, dy)) not in elves
            else:
                for dx in range(-1, 2):
                    ok &= (np + Point.of(dx, 0)) not in elves

            if ok:
                prop[np] += 1
                mov[elf] = np
                break
        else:
            #print(elf, "no move")
            mapp = {elf: "@" for elf in elves}
            mapp[elf] = "#"
            #print_coords(mapp)

    anym = False
    for elf, np in mov.items():
        if prop[np] == 1:
            elves.remove(elf)
            elves.add(np)
            anym = True

    if not anym:
        prints(round+1)
        exit()

    #print_coords(elves)

xs, ys = zip(*elves)
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)


for x in range(min_x, max_x + 1):
    for y in range(min_y, max_y+1):
        res += Point.of(x, y) not in elves


prints(res)
