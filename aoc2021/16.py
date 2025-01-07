#! /usr/bin/env pypy3
# 27/22
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("py", "in"))


DEC = '''0 = 0000
1 = 0001
2 = 0010
3 = 0011
4 = 0100
5 = 0101
6 = 0110
7 = 0111
8 = 1000
9 = 1001
A = 1010
B = 1011
C = 1100
D = 1101
E = 1110
F = 1111'''.split('\n')

D = {}
for l in DEC:
    a, b = l.split(' = ')
    D[a] = b

I = ''.join(D[v] for v in input())

packets = []
S = []

def f(i):
    V = int(I[i:i+3], 2)
    i += 3
    ID = int(I[i:i+3], 2)
    i += 3

    if ID == 4:
        s = []
        while True:
            s.append(I[i+1:i+5])
            if I[i] == '0': break
            i += 5

        i += 5

        packets.append((V, ID, int(''.join(s), 2)))
        S.append(packets[-1][-1])
    else:
        lid = int(I[i])
        i += 1


        if lid == 0:
            le = int(I[i:i+15], 2)
            i += 15

            start = i
            cnt = 0
            while i - start < le:
                i = f(i)
                cnt += 1
        else:
            cnt = int(I[i:i+11], 2)
            i += 11

            for _ in range(cnt):
                i = f(i)

        packets.append((V, ID, cnt))

        sub = [S.pop() for _ in range(cnt)]
        sub.reverse()

        if ID == 0:
            r = sum(sub)
        elif ID == 1:
            r = 1
            for v in sub: r *= v
        elif ID == 2:
            r = min(sub)
        elif ID == 3:
            r = max(sub)
        elif ID == 5:
            r = int(sub[0] > sub[1])
        elif ID == 6:
            r = int(sub[0] < sub[1])
        elif ID == 7:
            r = int(sub[0] == sub[1])

        S.append(r)

    return i

f(0)

print(sum(v for v, _, _ in packets))
prints(S.pop())



