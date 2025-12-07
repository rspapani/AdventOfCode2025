from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d0.txt")
raws = file.read().splitlines()
file.close()

rws = """
""".splitlines()

def proc(x):
    pass

inpt = lmap(proc, raws)

def f1(li):
    pass

def f2(li):
    pass

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
