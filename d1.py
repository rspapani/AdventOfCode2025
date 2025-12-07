from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d1.txt")
raws = file.read().splitlines()
file.close()

rasws = """L68
L30
R48
L5
R60
L55
L1
L99
R14
L82""".splitlines()

def proc(x):
    return ((-1) ** (x[0] == 'L'), int(x[1:]))

inpt = lmap(proc, raws)

def f1(li):
    pos = 50
    outs = 0

    for d, n in li:
        pos += (d*n)
        pos %= 100

        outs += pos == 0

    return outs

def f2(li):
    pos = 50
    outs = 0

    pi = 50
    for d, n in li:
        
        outs += n // 100
        n %= 100

        pos += (d*n)
        
        if (pi and pos <= 0) or pos >= 100:
            outs += 1

        pos %= 100
        pi = pos

    return outs

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
