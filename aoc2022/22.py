#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


map, passw = sys.stdin.read().split('\n\n')

map = [list(l) for l in lines(map)]
H = len(map)

instr = ['']
for c in passw.strip():
    if c in 'LR':
        instr.append(c)
        instr.append('')
    else:
        instr[-1] += c

W = max(len(l) for l in map)

print(instr)
#exit()

for x, c in enumerate(map[0]):
    if c == '.':
        p = Point.of(x, 0)
        break


CUBSIZE = 4 if len(instr) < 100 else 50
used = set()
cube_faces = []
for y, l in enumerate(map):
    for x, c in enumerate(l):
        if c != ' ' and (x, y) not in used:
            face = []
            for dy in range(CUBSIZE):
                line = []
                for dx in range(CUBSIZE):
                    nx, ny = x + dx, y + dy
                    assert map[ny][nx] != ' ' and (nx, ny) not in used
                    used.add((nx, ny))
                    line.append(map[ny][nx])
                face.append(line)

            cube_faces.append((face, x // CUBSIZE, y // CUBSIZE))

print([(x, y) for _, x, y in cube_faces])
print_coords({(x, y): str(i) for i, (_, x, y) in enumerate(cube_faces)})

ccoord_to_face = {}
for i, (_, x, y) in enumerate(cube_faces):
    ccoord_to_face[(x, y)] = i

left_rotate = rotate([[(x, y) for x in range(CUBSIZE)] for y in range(CUBSIZE)], times=3)

# Manually filled by looking at a folded cut-out of my input...
# RSWN
sw_face= [
# 0
[(1, 0), (2, 0), (3, 2), (5, 3)],
# 1
[(4, 2), (2, 3), (0, 0), (5, 0)],
# 2
[(1, 1), (4, 0), (3, 1), (0, 0)],
# 3
[(4, 0), (5, 0), (0, 2), (2, 3)],
# 4
[(1, 2), (5, 3), (3, 0), (2, 0)],
# 5
[(4, 1), (1, 0), (0, 1), (3, 0)],
]

pos = (Point.of(0, 0), 0)
print(len(instr))

facing = 0
dirs = list(Point.of(*d) for d in DIR)

for ins in instr:
    #print(pos, ins, facing)
    if ins.isdigit():
        steps = int(ins)
        for _ in range(steps):
            p, cface = pos
            np = p + dirs[facing]
            if 0 <= np.x < CUBSIZE and 0 <= np.y < CUBSIZE:
                nfacing = facing
                nface = cface
            else:
                np.x %= CUBSIZE
                np.y %= CUBSIZE

                nface, lrots = sw_face[cface][facing]
                nfacing = (facing - lrots) % 4

                for _ in range(lrots):
                    np = Point.of(*left_rotate[np.y][np.x])

            """
            np.x %= W
            np.y %= H
            #print(p, np, dirs[facing], ins)
            while np.x >= len(map[np.y]) or map[np.y][np.x] == ' ':
                np += dirs[facing]
                np.x %= W
                np.y %= H
            """

            c = cube_faces[nface][0][np.y][np.x]
            if c == "#":
                break
            else:
                pos = np, nface
                facing = nfacing
                #p = np
    else:
        facing = (facing + (1 if ins == 'R' else -1)) % 4


print(pos, facing)

p, cface = pos
_, cx, cy = cube_faces[cface]

fin_x = cx * CUBSIZE + p.x + 1
fin_y = cy * CUBSIZE + p.y + 1

res = fin_y * 1000 + fin_x * 4
res += facing


prints(res)
