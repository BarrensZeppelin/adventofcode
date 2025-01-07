#! /usr/bin/env pypy3
import z3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_z3.py", ".in"))


def max_geodes(time: int, blueprint: str) -> int:
    _, r1_cost, r2_cost, r3_cost_ore, r3_cost_clay, r4_cost_ore, r4_cost_obs = ints(
        blueprint
    )

    vars = [
        {
            var: z3.Int(f"{var}_{i}")
            for var in "ore clay obs r1 r2 r3 b1 b2 b3 b4".split()
        }
        for i in range(time + 1)
    ]

    s = z3.Optimize()
    for var in "ore clay obs r2 r3".split():
        s.add(vars[0][var] == 0)

    s.add(vars[0]["r1"] == 1)

    score = 0

    for t in range(time):
        vs = vars[t]
        if t:
            pvs = vars[t - 1]
            for i in range(1, 4):
                s.add(vs[f"r{i}"] == pvs[f"r{i}"] + pvs[f"b{i}"])

            s.add(
                vs["ore"]
                == pvs["ore"]
                + pvs["r1"]
                - (
                    pvs["b1"] * r1_cost
                    + pvs["b2"] * r2_cost
                    + pvs["b3"] * r3_cost_ore
                    + pvs["b4"] * r4_cost_ore
                )
            )

            s.add(vs["clay"] == pvs["clay"] + pvs["r2"] - (pvs["b3"] * r3_cost_clay))
            s.add(vs["obs"] == pvs["obs"] + pvs["r3"] - (pvs["b4"] * r4_cost_obs))

        bsum = 0
        for i in range(1, 5):
            v = vs[f"b{i}"]
            s.add(0 <= v, v <= 1)
            bsum += v

        s.add(bsum <= 1)

        for i, ore_cost in enumerate(
            (r1_cost, r2_cost, r3_cost_ore, r4_cost_ore), start=1
        ):
            s.add(vs[f"b{i}"] * ore_cost <= vs["ore"])

        s.add(vs["b3"] * r3_cost_clay <= vs["clay"])
        s.add(vs["b4"] * r4_cost_obs <= vs["obs"])

        score += vs["b4"] * (time - t - 1)

    sc = s.maximize(score)

    assert s.check() == z3.sat
    # return s.model().eval(score).as_long()
    return sc.value().as_long()


blueprints = lines()

res = sum(id * max_geodes(24, b) for id, b in enumerate(blueprints, start=1))
print(f"Part 1: {res}")

res = math.prod(max_geodes(32, b) for b in blueprints[:3])
print(f"Part 2: {res}")
