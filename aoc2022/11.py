#! /usr/bin/env pypy3
# 25/71
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace('py', 'in'))


monkeys = sys.stdin.read().split('\n\n')

def do_op(op: str, old: tuple[int]) -> tuple[int]:
    return tuple(
            eval(op, globals={"old": io}) % m[2]
            for io, m in zip(old, mon))

mon = []
for i, monk in enumerate(monkeys):
    _, st, op, t, ift, iff = lines(monk)

    items = list(ints(st))
    mon.append((
        [tuple([x] * 8) for x in items],
        op.split(": ")[1].split("new = ")[1],
        next(ints(t)),
        next(ints(ift)),
        next(ints(iff))))
    print(mon[-1])

    #print(st, op, t, ift, iff)

throws = [0] * len(mon)
for rnd in range(10**4):
    if rnd % 10 == 0:
        print(rnd)
    for i, (st, op, t, ift, iff) in enumerate(mon):
        assert ift != i and iff != i
        its = st[::]
        st.clear()
        for it in its:
            throws[i] += 1
            worry = do_op(op, it) #// 3

            if worry[i] % t == 0:
                mon[ift][0].append(worry)
            else:
                mon[iff][0].append(worry)


throws.sort(reverse=True)
print(throws)
print(throws[0] * throws[1])

