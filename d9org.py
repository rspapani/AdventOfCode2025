from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par, cache
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d9.txt")
raws = file.read().splitlines()
file.close()

rsaws = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3
""".splitlines()


def proc(x):
    a,b = lmap(int, x.split(','))
    return a + (b * 1j)

inpt = lmap(proc, raws)
area = lambda x: (abs(x.real) + 1) * (abs(x.imag) + 1)

def f1(li):
    return max(area(a - b)
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:]))

miny = min(a.imag for a in inpt) - 1
bounds = lzip(inpt, inpt[1:] + inpt[:1])

@cache
def contains(x, y, z, incl):
    mn, mx = sorted((x, y))
    return mn <= z <= mx - incl

@cache
def collides(a, b, c, d, incl):    
    vert1 = a.real == b.real
    vert2 = c.real == d.real

    if vert1 == vert2:
        return False

    elif vert1:
        cross1 = contains(c.real, d.real, a.real, incl)
        cross2 = contains(a.imag, b.imag, c.imag, incl)
        return cross1 and cross2

    else:
        return collides(c, d, a, b, incl)
    
@cache
def collisions(a, b, incl = False): 
    return sum(collides(x, y, a, b, incl)
               for x, y in bounds)

@cache
def interior(a, b, scope = 1):
    d = b - a
    dx = d.real
    dy = d - dx

    adx = (dx // abs(dx) if dx else 0) * scope
    ady = (dy.imag // abs(dy.imag) if dy.imag else 0) * 1j * scope

    p1 = a + adx + ady
    p2 = (a + ady) + (dx - adx)
    p3 = (a + adx) + (dy - ady)
    p4 = b - adx - ady

    return ((p1, p2), (p1, p3),
            (p2, p4), (p3, p4))

uniform = lambda a, b: not any(collisions(na, nb) for na, nb in interior(a, b, 1))

inside = lambda x: collisions(x, x.real + miny*1j, True) % 2 == 1

valid = lambda a, b: inside(interior(a, b, 1)[0][0]) and uniform(a, b)

def f2(li):
    return max(area(a - b)
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:])
               if valid(a, b)
               )

print("Part 1: ", int(f1(inpt)))
print("Part 2: ", int(f2(inpt)))

# a = 2 + 3j
# b = 9 + 5j

# a = 5467+67420j
# b = 94651+50319j

# print(a, b)

# print(inside(interior(a, b, 1)[0][0]))

# print(uniform(a, b))

# print(valid(a, b))

# print(area(b - a))


# print(interior(a, b))

# # for x in list((collides(x, y, na, nb), x, y, na, nb)
# #             for na, nb in interior(a, b)
# #             for x, y in bounds):
# #     if x[0]:
# #         print(x) 

# wrong answers
# 1525207666
# 1525135584
# 1525063500