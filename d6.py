from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d6.txt")
raws = file.read().splitlines()
file.close()

rasws = """
123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   + 
""".splitlines()[1:]

def proc(x):
    return lmap(int, x.split())

inpt = lmap(proc, raws[:-1]) + [raws[-1].split()]

ops = {'*' : lambda a, b: a * b,
       '+' : lambda a, b: a + b}

def f1(li):
    cols = zip(*li)
    return sum(reduce(ops[col[-1]], 
                      col[:-1]) 
               for col in cols)

raws2 = lzip(*raws[:-1])

def proc2(x):
    isint = lambda a: a in '1234567890'
    return ''.join(filter(isint, x))

inpt2 = lmap(proc2, raws2)

def f2(li, bttm):
    cols = [[]]
    for col in li:
        if col:
            cols[-1].append(int(col))
        else:
            cols.append([])

    return sum(reduce(ops[bttm[i]], 
                      col) 
               for i, col in enumerate(cols))

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt2, inpt[-1]))
