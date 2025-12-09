from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d8.txt")
raws = file.read().splitlines()
file.close()

rws = """162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689""".splitlines()

def proc(x):
    return vec(map(int, x.split(',')))

inpt = lmap(proc, raws)

def f1(li):
    pass

def f2(li):
    pass

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
