#! /usr/bin/env pypy3
from __future__ import annotations

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


dirs = list(Point.of(*d) for d in DIR)
map, passw = sys.stdin.read().split("\n\n")

map = [list(l) for l in lines(map)]
H = len(map)
W = max(len(l) for l in map)

instr = [""]
for c in passw.strip():
    if c in "LR":
        instr.extend((c, ""))
    else:
        instr[-1] += c


def part1():
    for x, c in enumerate(map[0]):
        if c == ".":
            p = Point.of(x, 0)
            break
    else:
        assert False

    facing = 0
    for ins in instr:
        if ins.isdigit():
            for _ in range(int(ins)):
                np = p + dirs[facing]
                np.x %= W
                np.y %= H

                while np.x >= len(map[np.y]) or (c := map[np.y][np.x]) == " ":
                    np += dirs[facing]
                    np.x %= W
                    np.y %= H

                if c == ".":
                    p = np
                else:
                    break
        else:
            facing = (facing + (1 if ins == "R" else -1)) % 4

    return p.x, p.y, facing


def score(x, y, d) -> int:
    return (y + 1) * 1000 + (x + 1) * 4 + d


print(score(*part1()))


def part2():
    CUBSIZE = 4 if len(instr) < 100 else 50
    used = set()
    cube_faces = []
    for y, l in enumerate(map):
        for x, c in enumerate(l):
            if c != " " and (x, y) not in used:
                face = []
                for dy in range(CUBSIZE):
                    line = []
                    for dx in range(CUBSIZE):
                        nx, ny = x + dx, y + dy
                        assert map[ny][nx] != " " and (nx, ny) not in used
                        used.add((nx, ny))
                        line.append(map[ny][nx])
                    face.append(line)

                cube_faces.append((face, x // CUBSIZE, y // CUBSIZE))

    NUM_FACES = len(cube_faces)

    # Pre-compute destinations after a single left rotation
    left_rotate = rotate(
        [[(x, y) for x in range(CUBSIZE)] for y in range(CUBSIZE)], times=3
    )

    def compute_swface() -> list[list[tuple[int, int]]]:
        """
        Compute destination face and number of left rotations to perform when
        leaving each face in each direction.
        """

        sw: list[list[tuple[int, int] | None]] = [
            [None] * 4 for _ in range(len(cube_faces))
        ]

        ccoord_to_face = {(x, y): i for i, (_, x, y) in enumerate(cube_faces)}

        # Fill in the initial constraints
        for (cx, cy), ci in ccoord_to_face.items():
            for fi, (dx, dy) in enumerate(dirs):
                nx, ny = cx + dx, cy + dy
                if (cj := ccoord_to_face.get((nx, ny))) is not None:
                    sw[ci][fi] = (cj, 0)

        class Done(Exception):
            pass

        def f():
            if all(None not in l for l in sw):
                # Check if the solution is valid
                for ci, l in enumerate(sw):
                    for fi in range(4):
                        # Do we return to the same state if we walk forwards 4 times?
                        i, cf = ci, fi
                        for _ in range(4):
                            i, lrots = sw[i][cf]
                            cf = (cf - lrots) % 4

                        if (i, cf) != (ci, fi):
                            return

                        # Do we return to the same state if walk forwards and turn 3 times?
                        for _ in range(3):
                            i, lrots = sw[i][cf]
                            cf = (cf - lrots - 1) % 4

                        if (i, cf) != (ci, fi):
                            return

                raise Done

            for ci, l in enumerate(sw):
                used = set(p[0] for p in l if p is not None) | {ci}
                for fi in range(4):
                    if l[fi] is None:
                        # Try all combinations of destination face and #left rotations
                        for cj, lrots in product(
                            set(range(NUM_FACES)) - used, range(4)
                        ):
                            nfi = (fi - lrots + 2) % 4

                            if sw[cj][nfi] is None:
                                sw[ci][fi] = (cj, lrots)
                                sw[cj][nfi] = (ci, -lrots % 4)

                                f()

                                sw[ci][fi] = sw[cj][nfi] = None

                        return

        try:
            f()
            assert False
        except Done:
            return sw  # type: ignore

    sw_face = compute_swface()
    p, cface, facing = Point.of(0, 0), 0, 0
    for ins in instr:
        if ins.isdigit():
            for _ in range(int(ins)):
                np = p + dirs[facing]
                if 0 <= np.x < CUBSIZE and 0 <= np.y < CUBSIZE:
                    nfacing, nface = facing, cface
                else:
                    np.x %= CUBSIZE
                    np.y %= CUBSIZE

                    nface, lrots = sw_face[cface][facing]
                    nfacing = (facing - lrots) % 4

                    for _ in range(lrots):
                        np = Point.of(*left_rotate[np.y][np.x])

                if cube_faces[nface][0][np.y][np.x] == ".":
                    p, cface, facing = np, nface, nfacing
                else:
                    break
        else:
            facing = (facing + (1 if ins == "R" else -1)) % 4

    _, cx, cy = cube_faces[cface]

    fin_x = cx * CUBSIZE + p.x
    fin_y = cy * CUBSIZE + p.y
    return fin_x, fin_y, facing


print(score(*part2()))
