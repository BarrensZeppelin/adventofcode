#! /usr/bin/env pypy3
from util import *

if len(sys.argv) == 1:
    sys.stdin = open(__file__.replace("_post.py", ".in"))

FS = sys.stdin.read()
secs = FS.split('inp w\n')[1:]
secs = [[s.split() for s in lines(sec)] for sec in secs]
N = len(secs)
assert N == 14

generated_code = ''
for i, sec in enumerate(secs):
	opt = []
	for instr, x, y in sec:
		if instr == 'add':
			f = '+'
		elif instr == 'mul':
			f = '*'
		elif instr == 'div':
			f = '//'
		elif instr == 'mod':
			f = '%'
		elif instr == 'eql':
			opt.append(f'{x} = int({x} == {y})')
			continue
		else:
			assert False, instr

		opt.append(f'{x} {f}= {y}')

	generated_code += f'def f_{i}(w, z, x=0, y=0):\n'
	opt.append('return z')
	generated_code += '\n'.join('\t' + s for s in opt) + '\n\n'

generated_code += 'funs = [' + ', '.join(f'f_{i}' for i in range(14)) + ']'
exec(generated_code)

cache = [set() for _ in range(14)]

def run(sec_no: int, pz: int) -> int:
    if sec_no == N: #or pz > 10 ** 7:
        return 0 if pz == 0 else -1

    if pz in cache[sec_no]: return -1

    for w in range(9, 0, -1):
    #for w in range(1, 10):
        nz = funs[sec_no](w, pz)
        r = run(sec_no + 1, nz)
        if r != -1:
            return r + w * 10 ** (N - sec_no - 1)

    cache[sec_no].add(pz)
    return -1

print(run(0, 0))
