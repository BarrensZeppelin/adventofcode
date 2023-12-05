#!/usr/bin/env pypy3
# 32/296

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

seeds, *blocks = sys.stdin.read().split("\n\n")
seeds = list(ints(seeds.split(": ")[1]))
seeds = [t for t in tile(seeds, 2)]
print(seeds)
# exit()

maps = []
for block in blocks:
    ls = lines(block)[1:]
    maps.append([tuple(ints(l)) for l in ls])

for l in maps:
    print(l)

best = 1<<60
def f(i: int, l: int, r: int):
    global best
    if r <= l: return
    if i >= len(maps):
        best = min(best, l)
        return

    for sd, ss, llen in maps[i]:
        shift = sd - ss
        sr = ss + llen
        sdr = sd + llen
        if r <= ss or sr <= l:
            continue

        if l < ss:
            assert r >= ss
            f(i, l, ss)
        if r > sr:
            assert l <= sr
            f(i, sr, r)

        l, r = max(l, ss), min(r, sr)
        f(i+1, l+shift, r+shift)
        break
    else:
        f(i+1, l, r)

for seed, seedlen in seeds:
    f(0, seed, seed+seedlen)

prints(best)
exit()

for l in maps:
    nseeds = []
    def add(l, len):
        assert l
        if len > 0:
            nseeds.append((l, len))
    for seed, seedlen in seeds:
        print("considering", seed, seedlen)
        for sd, ss, len in l:
            if seed + seedlen <= ss or ss + len <= seed:
                continue

            if ss <= seed and seed + seedlen <= ss + len:
                add(sd + (seed - ss), seedlen)
            elif seed <= ss and ss + len <= seed + seedlen:
                seeds.append((seed, ss - seed))
                add(sd, len)
                seeds.append((ss+len, seed + seedlen - ss - len))
            elif seed <= ss:
                seeds.append((seed, ss - seed))
                add(sd, (seed+seedlen-ss-len))
            else:
                assert ss + len <= seed + seedlen
                add(seed, ss+len-seed)
                seeds.append((ss+len, seed + seedlen - ss - len))

            #
            # add(seed, ss - seed)
            # start = max(seed, ss)
            # print(f"{sd=} {ss=} {len=} {seed=} {seedlen=} {start=}")
            # add(sd + start - ss, min(len, seedlen))
            # if seed + seedlen > ss + len:
            #     add(ss+len, seed + seedlen - ss - len)
            break
        else:
            print("no match!")
            add(seed, seedlen)

    seeds = nseeds
    print("seeds", seeds)
# for seed in seeds:
#
#     best = min(best, seed)


prints(min(s[0] for s in seeds))

# prints(best)
