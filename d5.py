from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d5.txt")
raws = file.read()
file.close()

rawss = """3-5
10-14
16-20
12-18

1
5
8
11
17
32
"""

raw1, raw2 = lmap(lambda x: x.splitlines(), raws.split("\n\n"))

def proc(x):
    return lmap(int, x.split('-'))

inpt = lmap(sorted, (lmap(proc, raw1), lmap(int, raw2)))

def f1(li):
    ranges, vals = li
    outs = 0
    ri = 0

    for val in vals:
        a, b = ranges[ri]

        while b < val and ri < len(ranges) - 1:
            ri += 1
            a, b = ranges[ri]

        outs += (a <= val <= b)

    return outs

collide = lambda a, b: b[0] <= a[1]
compact = lambda a, b: (a[0], max(a[1], b[1]))

def f2(li):
    ranges, _ = li
    outs = [ranges[0]]

    for range in ranges[1:]:
        if collide(outs[-1], range):
            outs[-1] = compact(outs[-1], range)

        else:
            outs.append(range)
    
    return sum(b + 1 - a for a,b in outs)

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
