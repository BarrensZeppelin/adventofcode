#!/usr/bin/env pypy3
# 99/794

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

kind = {}
ADJ = {}

for l in lines():
    name, adj = l.split(" -> ")
    if name[0] in "%&":
        k = name[0]
        name = name[1:]
    else:
        k = "b"
    kind[name] = k
    ADJ[name] = adj.split(", ")

state = {n: False for n in kind}
cst = {n: {} for n in kind if kind[n] == "&"}
for name, a in ADJ.items():
    for b in a:
        if kind.get(b) == "&":
            cst[b][name] = False

# for name, d in cst.items():
#     print(len(d), name, d)
RADJ = defaultdict(list)
for a, b in ADJ.items():
    for c in b:
        RADJ[c].append(a)

GC = RADJ["zh"]

# _, RV, _ = bfs(RADJ, "rx")
# print(f"{RV=}")
# exit()

T = [0, 0]
prevstate = dict(state)
ST = defaultdict(list)
nums = {}
valid = set()
for press in range(2**12):
    # if press % (10**4) == 0:
    #     print(press)

    # prevc = {n: dict(cst[n]) for n in cst}
    for a, b in state.items():
        if kind[a] == "%":
            ST[a].append(int(b))

    Q = [("", "broadcaster", False)]
    for frm, name, typ in Q:
        T[typ] += 1

        if name == "rx" and not typ:
            prints(press+1)
            exit()

        k = kind.get(name, "output")
        if name == "broadcaster":
            for a in ADJ[name]:
                Q.append((name, a, typ))
        elif k == "%":
            if not typ:
                s = not state[name]
                state[name] = s
                for a in ADJ[name]:
                    Q.append((name, a, s))
        elif k == "&":
            # assert kind[name] == "&", (name, kind[name])
            d = cst[name]
            d[frm] = typ
            s = not all(d.values())
            for a in ADJ[name]:
                Q.append((name, a, s))

    for name, _, typ in Q:
        if name in GC and typ:
            if name not in nums:
                nums[name] = press+1
            else:
                assert (press+1) % nums[name] == 0

assert len(nums) == 4
prints(math.prod(nums.values()))
    # print(press, {a: int(b) for a, b in state.items() if kind[a] == "%"})
"""
LS = []
for a, l in ST.items():
    s = "".join(map(str, l))
    for pw in range(1, 100):
        xx = 2 ** pw
        lenl = len(l)
        if xx < lenl and 1 in l[:xx] and s == s[:xx] * (lenl // xx):
            LS.append((pw, a))
            break
    else:
        print("NO", a, state[a])

    # LS.append(("".join(map(str, l)), a))
LS.sort()
for a, b in LS:
    print(a, b)
"""
# print(ST)

# for a, b in prevstate.items():
#     if b != state[a]:
#         print(f"{press=} {a=} {b=} {state[a]=}")
# print(cst)
# a, b = T
# print(T)

# prints(a * b)
