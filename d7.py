from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d7.txt")
raws = file.read().splitlines()
file.close()

rsaws = """.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............""".splitlines()

def proc(x):
    return [ch != '.' for ch in x]

inpt = lmap(proc, raws)

def f1(li):
    n = len(li[0])
    curr = li[0][:]
    outs = 0

    for row in li[1:]:
        nx = [0] * n
        for pos, par in enumerate(zip(curr, row)):
            a,b = par

            if a and b:
                outs += 1
                nx[pos] = 0

                if pos and not row[pos - 1]:
                    nx[pos-1] |= 1

                if pos < n - 1 and not row[pos + 1]:
                    nx[pos+1] |= 1

            elif a:
                nx[pos] = 1

        curr = nx

    return outs

def f2(li):
    n = len(li[0])
    curr = li[0][:]

    map = []

    for row in li[1:]:
        nx = [0] * n
        for pos, par in enumerate(zip(curr, row)):
            a,b = par

            if a and b:
                nx[pos] = 0

                if pos and not row[pos - 1]:
                    nx[pos-1] += a

                if pos < n - 1 and not row[pos + 1]:
                    nx[pos+1] += a

            elif a:
                nx[pos] += a

        curr = nx
    #     print(curr)
    #     map.append("".join('|' if x else '.' for x in curr))

    # print('\n'.join(map))

    return sum(curr)

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
