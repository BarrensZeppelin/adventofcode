#! /usr/bin/env pypy3
# 16/115
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))

MIN = 26

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

for a in adj:
    for b in adj[a]:
        assert b in adj, b

nonz = {a for a, rate in rates.items() if rate > 0}
dists = {a: bfs(adj, a)[0] for a in adj}

nonzi = {a: i for i, a in enumerate(nonz)}


D = defaultdict(lambda: -1)
D[(0, "AA", "AA", 0)] = 0
V = set()
#Q = [(0, 0, "AA", "AA", 0)]
Q = [set() for _ in range(MIN+1)]
Q[0].add(("AA", "AA", 0))
#heapify(Q)
best = 0
def edge(ntime, nrate, a1, a2, nnonz):
    if a1 > a2: a1, a2 = a2, a1
    if ntime <= MIN and nrate > D[(ntime, a1, a2, nnonz)]:
        D[(ntime, a1, a2, nnonz)] = -nrate #(ntime, nrate)
        Q[ntime].add((a1, a2, nnonz))
        #heappush(Q, (ntime, -nrate, a1, a2, nnonz))

FULL = (1 << len(nonz))-1

for time in range(MIN+1):
    print(time, len(Q[time]))
    for a1, a2, opened in Q[time]:
        #time, rate, a1, a2, opened = heappop(Q)
        rate = -D[(time, a1, a2, opened)]
        best = max(best, rate)
        if time == MIN:
            prints(rate)
            break
        if opened == FULL: continue

        # I should've tried this heuristic earlier...
        estimate = 0
        for b, i in nonzi.items():
            if not opened & (1 << i):
                # Writing |= here instead of += was a typo.
                # It definitely makes the heuristic invalid, I was just lucky
                # that it didn't result in a WA on my input.
                estimate |= (MIN-time-1) * rates[b]

        if rate + estimate < best: continue

        for b1 in (adj[a1] + [a1]):
            for b2 in (adj[a2] + [a2]):
                b1i = nonzi.get(b1, 0)
                b2i = nonzi.get(b2, 0)
                edge(time+1, rate, b1, b2, opened)

                #print(MIN-time-2, opened, b1, b2, rates[b1], rates[b2])
                if b1 in nonz and not opened & (1 << b1i): # not in opened:
                    nrate = rate + (MIN-time-2) * rates[b1]
                    if a1 == b1:
                        edge(time+1, nrate+rates[b1], b1, b2, opened | (1 << b1i))

                    for c2 in adj[b2] + [b2]:
                        edge(time+2, nrate, b1, c2, opened | (1 << b1i))

                    if b1 != b2 and b2 in nonz and not opened & (1 << b2i):# not in opened:
                        nrate = rate + (MIN-time-2) * (rates[b1] + rates[b2])
                        edge(time+2, nrate, b1, b2, opened | (1 << b1i) | (1 << b2i))
                if b2 in nonz and not opened & (1 << b2i): # not in opened:
                    nrate = rate + (MIN-time-2) * (rates[b2])
                    if a2 == b2:
                        edge(time+1, nrate+rates[b2], b1, b2, opened | (1 << b2i))

                    for c1 in adj[b1] + [b1]:
                        edge(time+2, nrate, c1, b2, opened | (1 << b2i))

        """
        for b in nonz:
            if b not in opened:
                d = dists[a][b]
                ntime = time + d + 1
                nrate = rate + (MIN-ntime) * rates[b]
                nnonz = frozenset(set(opened) | {b})
                #edge(ntime, nrate,
                if ntime <= MIN and (ntime, b, nnonz) not in V and nrate > D[(ntime, b, nnonz)]:
                    D[(ntime, b, nnonz)] = nrate #(ntime, nrate)
                    heappush(Q, (ntime, -nrate, b, nnonz))
        """

    Q[time].clear()
print(best)
