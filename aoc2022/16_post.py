#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

adj = defaultdict(list)
rates = {}

for line in lines():
    parts = line.split()
    a = parts[1]
    rate = next(ints(line))
    rates[a] = rate
    if "valves" in line:
        for b in line.split("valves ")[-1].split(", "):
            adj[a].append(b)
    else:
        adj[a].append(parts[-1])

nonz = {a for a, rate in rates.items() if rate > 0}
dists = {a: bfs(adj, a)[0] for a in adj}


def calc(Time, make_edges):
    D = [defaultdict(int) for _ in range(Time + 1)]
    D[0][("AA", "AA", frozenset())] = 0

    best = 0

    def edge(ntime, nrate, a1, a2, nnonz, to_open):
        nonlocal best
        if ntime > Time:
            return
        if to_open:
            nrate += (Time - ntime) * sum(rates[b] for b in to_open)
            nnonz |= to_open
        if nrate > D[ntime][(a1, a2, nnonz)]:
            best = max(best, nrate)
            D[ntime][(a1, a2, nnonz)] = nrate

    pruned = 0
    for time in range(Time + 1):
        for (a1, a2, opened), rate in D[time].items():
            if time == Time:
                break

            heuristic = rate
            for b in set(nonz) - opened:
                dist = min(dists[b][a1], dists[b][a2])
                heuristic += max(0, (Time - time - dist - 1) * rates[b])

            if heuristic <= best:
                pruned += 1
                continue

            make_edges(time, a1, a2, opened, rate, edge)

    print("Pruned", pruned)
    return best


def part1(time, a1, a2, opened, rate, edge):
    for b in nonz - opened:
        edge(time + dists[a1][b] + 1, rate, b, a2, opened, {b})


print(calc(30, part1))


def part2(time, a1, a2, opened, rate, _edge):
    def edge(b1, b2, to_open):
        # Halve search space
        if b1 > b2:
            b1, b2 = b2, b1
        _edge(time + 1, rate, b1, b2, opened, to_open)

    for b1 in adj[a1] + [a1]:
        for b2 in adj[a2] + [a2]:
            edge(b1, b2, set())

            if a1 == b1 and b1 in nonz and b1 not in opened:
                edge(b1, b2, {b1})

                if a2 == b2 and b2 in nonz and b2 not in opened:
                    edge(b1, b2, {b1, b2})

            if a2 == b2 and b2 in nonz and b2 not in opened:
                edge(b1, b2, {b2})


# This runs in about 15 seconds
print(calc(26, part2))
