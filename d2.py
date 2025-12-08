from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d2.txt")
raws = file.read().splitlines()
file.close()

rawss = """11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124""".splitlines()

raws = raws[0].split(',')

def proc(x):
    return tuple(map(int, x.split('-')))

inpt = lmap(proc, raws)

def isrep(y):
    x = str(y)
    n = len(x)//2
    return x[:n] == x[n:]

def f1(li):
    findreps = lambda x, y: sum(filter(isrep, range(x, y + 1)))
    return sum(findreps(x, y) for x, y in li)

def hasrep(y):
    x = str(y)
    n = len(x)
    
    for i in range(1, n//2):
        rhs = x[i:] + x[:i]

        if rhs == x:
            return True

    return False

def f2(li):
    findreps2 = lambda x, y: sum(filter(hasrep, range(x, y + 1)))
    return sum(findreps2(x, y) for x, y in li)

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
