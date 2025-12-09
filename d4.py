from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d4.txt")
raws = file.read().splitlines()
file.close()

rsaws = """..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.
""".splitlines()

def proc(x):
    return [ch == '@' for ch in x]

inpt = lmap(proc, raws)
m = len(inpt)
n = len(inpt[0])

def f1(li):
    return sum(
                sum(li[int(dp.real)][int(dp.imag)]
                   for dp in cadjs(y + (x * 1j))
                   if 0 <= dp.real < m and
                      0 <= dp.imag < n
                  ) < 4 
                for y, row in enumerate(li)
                for x, roll in enumerate(row)
                if roll
               )

def irremovable(li):
    return [
            [sum(li[int(dp.real)][int(dp.imag)]
                for dp in cadjs(y + (x * 1j))
                if 0 <= dp.real < m and
                    0 <= dp.imag < n
                ) >= 4 
                if roll else False
                for x, roll in enumerate(row)
            ]
            for y, row in enumerate(li)
            ]

cnt = lambda li: sum(map(sum, li))

def f2(li):
    og = curr = cnt(li)
    prev = 0

    while curr != prev:
        li = irremovable(li)
        prev, curr = curr, cnt(li)

    return og - curr

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
