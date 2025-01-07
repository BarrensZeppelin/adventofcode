#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0
P = Point

c = P.of(0, 0)
D = dict(zip("RULD", DIR_NORTHNEG))
HD = dict(zip("0123", "RDLU"))
points = [c]
for l in lines():
    d, a, hex = l.split()
    a = int(a)
    hex = hex[2:-1]
    a = int(hex[:5], 16)
    d = HD[hex[5]]

    c += P(D[d]) * a
    # for _ in range(a):
    #     c += D[d]
    points.append(c)

    # print(points[-1], c, d, a)
print(len(points))
assert points[0] == points[-1]

area = abs(sum(points[i-1].cross(points[i]) for i in range(len(points))))
sidelen = sum((points[i]-points[i-1]).manh_dist() for i in range(len(points)))
interior = area // 2 - sidelen // 2 + 1
prints(sidelen + interior)
exit()

sy = min(p.y for p in points)
ey = max(p.y for p in points)
for y in range(sy, ey+1):
    # ps = [p.x for p in points if p.y == y]
    ps = []
    for p1, p2 in windows(points, 2):
        if p1.y == p2.y == y:
            ps.append((min(p1.x, p2.x), max(p1.x, p2.x)))
        else:
            my, may = min(p1.y, p2.y), max(p1.y, p2.y)
            if my <= y < may:
                ps.append((p1.x, p1.x))

    ps = merge_intervals(ps)
    flag = False
    last = -1
    contr = 0
    for a, b in ps:
        if flag:
            contr += b - last + 1
        else:
            contr += b - a + 1
        last = b+1
        flag = not flag
        # res += p2 - p1+1
    res += contr
    print(y, contr, ps)
# print_coords(points)
prints(res)
exit()

# print(points[0], points[1], points[-1], points[-2])
# exit()
# for dp in OCTDIR:
#     p = points[0] + dp
#     if p in points:
#         continue
#
#     Q = [p]
#     V = set(Q) | set(points)
#     ok = True
#     for i in Q:
#         if len(V) >= 10 ** 6:
#             break
#         for j in map(P, neighbours(*i, dirs=DIR)):
#             if j not in V:
#                 V.add(j)
#                 Q.append(j)
#     else:
#         prints(len(V))
#         break

