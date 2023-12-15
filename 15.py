#!/usr/bin/env pypy3
# 27/209

from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))

res = 0

boxes = [[] for _ in range(256)]
lab_strength = {}

for seq in input().split(","):
    if "-" in seq:
        assert seq[-1] == "-"
        op = "-"
        seq = seq[:-1]
    else:
        assert "=" in seq
        seq, fl = seq.split("=")
        op = "="
        fl = int(fl)

    h = 0
    for c in seq:
        h = (h + ord(c)) * 17 % 256
    # res += h

    if op == "-":
        s = lab_strength.get(seq)
        if seq in boxes[h]:
            boxes[h].remove(seq)
    else:
        lab_strength[seq] = fl
        for i, oseq in enumerate(boxes[h]):
            if oseq == seq:
                boxes[h][i] = seq
                break
        else:
            boxes[h].append(seq)

for bi, b in enumerate(boxes, 1):
    if b:
        print(bi, b)



for bi, b in enumerate(boxes, 1):
    for li, seq in enumerate(b, 1):
        res += bi * li * lab_strength[seq]
        # lab_to_box[seq] = (bi, li)



prints(res)
