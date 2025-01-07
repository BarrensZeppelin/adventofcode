#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))


def max_geodes(time: int, blueprint: str) -> int:
    _, r1_cost, r2_cost, r3_cost_ore, r3_cost_clay, r4_cost_ore, r4_cost_obs = ints(
        blueprint
    )

    max_ore_cost = max(r2_cost, r3_cost_ore, r4_cost_ore)

    def g(i, o, c, obs, r1, r2, r3):
        "Bound the search space a little bit"

        rtime = time - i
        # We want to build at least one geode-cracking robot
        o = min(o, (rtime - 1) * max_ore_cost + r4_cost_ore)
        c = min(c, (rtime - 1) * r3_cost_clay)

        if r1 == max_ore_cost:
            o = min(o, max_ore_cost)

        if r2 == r3_cost_clay:
            c = min(c, r3_cost_clay)

        return f(i, o, c, obs, r1, r2, r3)

    @lru_cache(maxsize=None)
    def f(i, o, c, obs, r1, r2, r3):
        if i == time:
            return 0

        n_ore = o + r1
        n_clay = c + r2
        n_obs = obs + r3

        if o >= r4_cost_ore and obs >= r4_cost_obs:
            return g(
                i + 1,
                n_ore - r4_cost_ore,
                n_clay,
                n_obs - r4_cost_obs,
                r1,
                r2,
                r3,
            ) + (time - i - 1)

        best = g(i + 1, n_ore, n_clay, n_obs, r1, r2, r3)
        if o >= r3_cost_ore and c >= r3_cost_clay and r3 < r4_cost_obs:
            best = max(
                best,
                g(
                    i + 1,
                    n_ore - r3_cost_ore,
                    n_clay - r3_cost_clay,
                    n_obs,
                    r1,
                    r2,
                    r3 + 1,
                ),
            )
        else:
            if o >= r1_cost and r1 < max_ore_cost:
                best = max(
                    best,
                    g(i + 1, n_ore - r1_cost, n_clay, n_obs, r1 + 1, r2, r3),
                )

            if o >= r2_cost and r2 < r3_cost_clay:
                best = max(
                    best,
                    g(i + 1, n_ore - r2_cost, n_clay, n_obs, r1, r2 + 1, r3),
                )

        return best

    return f(0, 0, 0, 0, 1, 0, 0)


blueprints = lines()

res = sum(id * max_geodes(24, b) for id, b in enumerate(blueprints, start=1))
print(f"Part 1: {res}")

res = math.prod(max_geodes(32, b) for b in blueprints[:3])
print(f"Part 2: {res}")
