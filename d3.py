from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d3.txt")
raws = file.read().splitlines()
file.close()

sraws = """987654321111111
811111111111119
234234234234278
818181911112111
""".splitlines()

def proc(x):
    return x

inpt = lmap(proc, raws)

indmax = lambda x: (-1) * max((y, -i) for i, y in enumerate(x))[1]

def maxjolt(x):
    i1 = indmax(x[:-1])
    i2 = i1 + 1 + indmax(x[i1 + 1:])
    return int(x[i1] + x[i2])

def f1(li):
    return sum(map(maxjolt, li))

def maxnjolt(x, n = 12):
    if n == 1:
        return max(x)

    i1 = indmax(x[:1-n])
    return x[i1] + maxnjolt(x[i1 + 1:], n - 1)

def f2(li):
    return sum(int(x) for x in map(maxnjolt, li))

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
