#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

G = lines()

W, H = len(G[0]), len(G)
assert W == H
start = next((x, y) for y, l in enumerate(G) for x, c in enumerate(l) if c == "S")

get_adj = lambda G: grid_adj_2d(W, H, pred=lambda _, p: G[p[1]][p[0]] != "#")
print(f"""Part 1: {
    sum(d <= 64 and d % 2 == 0 for d in bfs(get_adj(G), start)[0].values())
}""")

steps = 26501365
LIM = ceildiv(steps, W)
res = 0

for mirrored, G in enumerate((G, [l[::-1] for l in G])):
    adj = get_adj(G)
    distances = {p: list(bfs(adj, p)[0].values()) for p in product((0, W // 2, W-1), repeat=2)}
    maxdist = max(max(d) for d in distances.values())

    @cache
    def reaches(p: tuple[int, int], steps: int) -> int:
        return sum(d <= steps and d % 2 == steps % 2 for d in distances[p])

    @cache
    def by_parity(p: tuple[int, int], parity: int) -> int:
        return sum(d % 2 == parity for d in distances[p])

    def tile_distance(diff: int) -> int:
        if not diff: return 0
        return W * diff - sign(diff) * (W // 2)

    X = 0
    for row in range(-LIM, LIM+1):
        row_dist = tile_distance(row)
        too_far = lambda x: abs(row_dist) + tile_distance(x) > steps

        while not too_far(X): X += 1
        while X >= 1 and too_far(X-1): X -= 1

        x = X-1
        while x >= mirrored:
            x_dist = tile_distance(x)
            remaining_steps = steps - abs(row_dist) - x_dist
            p = ((start[0] + x_dist) % W, (start[1] + row_dist) % H)

            if remaining_steps >= maxdist:
                # We can reach all plots in the tile (with the correct parity)
                if x > 2:
                    jump = (x - 1) // 2
                    res += len(distances[p]) * jump
                    x -= jump * 2
                else:
                    res += by_parity(p, remaining_steps % 2)
                    x -= 1
            else:
                res += reaches(p, remaining_steps)
                x -= 1


prints(f"Part 2: {res}")
