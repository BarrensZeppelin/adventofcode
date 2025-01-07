#!/usr/bin/env pypy3
# 103/19

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

for l in lines():
    l, counts = l.split()
    counts = list(ints(counts))
    counts *= 5
    l = "?".join(l for _ in range(5))
    N = len(l)
    M = len(counts)

    @lru_cache(maxsize=None)
    def dp(i, j):
        if i == N:
            return int(j == M)

        r = 0
        if l[i] in ".?":
            r += dp(i+1, j)

        if l[i] in "#?":
            if j < M:
                c = counts[j]
                if i + c <= N and "." not in set(l[i:i+c]) and (i + c == N or l[i+c] != "#"):
                    r += dp(min(i+c+1, N), j+1)
        return r





    res += dp(0, 0)
    continue

    poss = []
    for c in l:
        if c == "?":
            poss.append(".#")
        else:
            poss.append(c)


    for aa in product(*poss):
        if Counter(aa)["#"] != sum(counts):
            continue
        i = 0
        ok = True
        for count in counts:
            while i < N and aa[i] == ".":
                i += 1

            if i == N:
                ok = False
                break

            for x in range(count):
                if i == N or aa[i] != "#":
                    ok = False
                    break

                i += 1

            if i < N and aa[i] != ".":
                ok = False
                i += 1

            if not ok:
                break
        res += ok


prints(res)
