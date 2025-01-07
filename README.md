# Advent of Code

This repository contains my solutions and helper scripts for the [Advent of Code](https://adventofcode.com/) challenges from 2021 onwards.

To compete for global leaderboard positions, I use Python with [PyPy](https://www.pypy.org/).
Sometimes I use [Z3](https://github.com/Z3Prover/z3) through its Python bindings, and occasionally the [networkx](https://networkx.org/) library,
but mostly I use a small set of utility functions and classes that I've developed over the years.

| Year | Points | Rank |
|------|-------:|-----:|
| 2021 |   1874 |   28 |
| 2022 |   1902 |   24 |
| 2023 |   1992 |   16 |
| 2024 |    720 |   88 |

Most `[dd].py` files contain the un-edited solution that I used to solve the puzzle initially (very messy!).
In the interest of speed, the solution for part 1 is often clobbered to solve part 2 quickly.

There are also some `[dd]_post.py` file that contain cleaned-up versions of the solutions that solves both parts.

#### Prolog

I have also written some solutions in [SWI Prolog](https://www.swi-prolog.org/) for fun (not for the leaderboard).

They are run from the command line like so:
```bash
swipl -O -t halt -g main 13.pl < 13.in
```
