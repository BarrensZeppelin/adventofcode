#! /usr/bin/env python
from util import *

program = sys.stdin.read()

parts = re.findall(
    r"""inp w
mul x 0
add x z
mod x 26
div z (\d+)
add x (-?\d+)
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y (\d+)
mul y x
add z y""",
    program,
)

from z3 import *

o = Optimize()

cnt = Counter()
VARS = "zw"
vs = {(c, 0): Int(f"{c}_0") for c in VARS}
o.add(vs[("z", 0)] == 0)


def pv(x):
    a = vs[(x, cnt[x])]
    cnt[x] += 1
    b = Int(f"{x}_{cnt[x]}")
    vs[(x, cnt[x])] = b
    return a, b


"""
x = ((z % 26) + add_x != w)
z //= div_z

if x == 1:
	z = z * 26 + (w + add_y)
"""

objective = 0

for args in parts:
    div, addx, addy = map(int, args)

    _, w = pv("w")
    objective = objective * 10 + w
    o.add(And(1 <= w, w <= 9))

    pz, nz = pv("z")
    o.add(nz == If((pz % 26) + addx == w, pz / div, pz / div * 26 + w + addy))

o.add(nz == 0)

print(o)

o.set(priority="box")
max_id = o.maximize(objective)
min_id = o.minimize(objective)

assert o.check() == sat

print(max_id.value())
print(min_id.value())
