#!/usr/bin/env pypy3

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0
CNT = dict(zip("red green blue".split(), (12, 13, 14)))

for l in lines():
    # l = l.replace(";", ",")
    _, rest = l.split("ame ")
    id, rest = rest.split(": ", maxsplit=1)
    maxx = Counter()
    for reveal in rest.split("; "):
        print(reveal)
        used = Counter()
        for blk in reveal.split(", "):
            print(blk)
            cnt, type = blk.split(" ")
            maxx[type] = max(maxx[type], int(cnt))
            used[type] += int(cnt)
            # print(cnt, type)
    #     if any(v > CNT[t] for t, v in used.items()):
    #         break
    # else:
    #     res += int(id)

    res += math.prod(maxx.values())

prints(res)
