#! /usr/bin/env python
# 34/8
from util import *

sys.stdin = open(__file__.replace("_np.py", ".in"))

import numpy as np

C = Counter(ints())
V = [C[i] for i in range(9)]

A = np.eye(9, k=1, dtype=int)

A[6][0] = 1
A[8][0] = 1

def f(n: int) -> int:
    return sum(np.linalg.matrix_power(A, n) @ V)

print(f(80))
print(f(256))
