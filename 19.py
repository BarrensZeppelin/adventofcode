#! /usr/bin/env pypy3
# 15/4
#import z3
import gc
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


res = 1

for id, l in enumerate(lines(), start=1):
    _, ore, clay, oore, oclay, gore, gobs = ints(l)
    #if id == 1 or id == 2: continue
    if id == 4: break

    """
    vars = {
        var: [z3.Int(f"{var}_{i}") for i in range(25)]
        for var in "ore clay obs geo r1 r2 r3 r4".split()
    }


    s = z3.Optimize()
    for var in "ore clay obs r2 r3 r4".split():
        s.add(vars[var][0] == 0)

    s.add(vars["r1"][0] == 1)

    for t in range(1, 24):
    """


    @lru_cache(maxsize=None)
    def f(i, o, c, obs, r1, r2, r3, r4):
        if i == 32:
            return 0

        #print(i, o, c, obs, r1, r2, r3, r4)
        best = f(i+1, o+r1, c+r2, obs+r3, r1, r2, r3, r4)
        if o >= gore and obs >= gobs:
            best = max(best, f(i+1, o-gore+r1, c+r2, obs-gobs+r3, r1, r2, r3, r4+1))
        elif o >= oore and c >= oclay:
            best = max(best, f(i+1, o-oore+r1, c-oclay+r2, obs+r3, r1, r2, r3+1, r4))
        else:
            canb = False
            if o >= ore and r1 < max(gore, oore, clay):
                canb = True
                best = max(best, f(i+1, o-ore+r1, c+r2, obs+r3, r1+1, r2, r3, r4))

            if o >= clay and r2 < oclay:
                canb = True
                best = max(best, f(i+1, o-clay+r1, c+r2, obs+r3, r1, r2+1, r3, r4))

            #if not canb:
            #best = max(best, f(i+1, o+r1, c+r2, obs+r3, r1, r2, r3, r4))

        return best+r4


    print(id)
    lv = f(0, 0, 0, 0, 1, 0, 0, 0)
    #res += id * lv
    res *= lv
    print(id, lv)
    gc.collect()

prints(res)
