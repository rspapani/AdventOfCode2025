## Time Stats:
# (3.12.7) ➜ AdventOfCode2025 (main) ✗ time python d9.py
# Part 1:  4759930955
# Part 2:  1525241870
# python d9.py  0.80s user 0.04s system 92% cpu 0.904 total

# from collections import defaultdict as ddict, Counter as count
# from functools import reduce,  cmp_to_key, partial as par, cache
# from math import prod, sqrt as root, lcm as lcm, gcd as gcd
# from operator import itemgetter as ig
# from re import findall as rall

# import math
# import re

# from aoc import *

from functools import cache

lmap = lambda f, li: [f(x) for x in li]

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

def downsample(inpt, ratio=1):
    xs = sorted(set(p.real for p in inpt))
    ys = sorted(set(p.imag for p in inpt))
    
    xmap = {x: i*ratio for i, x in enumerate(xs)}
    ymap = {y: i*ratio for i, y in enumerate(ys)}
    
    xinv = {i*ratio: x for i, x in enumerate(xs)}
    yinv = {i*ratio: y for i, y in enumerate(ys)}
    
    downsampled = [xmap[p.real] + ymap[p.imag]*1j for p in inpt]
    
    return downsampled, xinv, yinv

inpt, xinv, yinv = downsample(inpt)
upsample = lambda a: xinv[a.real] + (yinv[a.imag] * 1j)
area_scaled = lambda a, b: area(upsample(b) - upsample(a))

def f1(li):
    return max(area_scaled(a,b)
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:]))

@cache
def points(a, b):
    if a == b:
        return (a,)

    d = b - a
    step = d/abs(d)

    return tuple(a + (x * step)
                 for x in range(int(abs(d)) + 1))

vert = lambda a, b: (b - a).real == 0

def splitbounds(bounds):
    vbounds, vbends = set(), set()
    hbounds, hbends = set(), set()

    for a, b in bounds:
        if vert(a, b):
            vbounds.update(points(a, b)[:-1])
            vbends.add(b)

        else:
            hbounds.update(points(a, b)[:-1])
            hbends.add(b)

    return vbounds, vbends, hbounds, hbends

miny = min(a.imag for a in inpt) - 1
bounds = zip(inpt, inpt[1:] + inpt[:1])
vbounds, vbends, hbounds, hbends = splitbounds(bounds)

@cache
def collisions(a, b, incl=True):
    if vert(a, b):
        dbounds = hbounds
        dbends = hbends
    else:
        dbounds = vbounds
        dbends = vbends

    checks = set(points(a, b))

    return len(dbounds & checks) + \
            (len(dbends & checks) if incl else 0)

@cache
def interior(a, b, scope = 0):
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

inside = lambda x: collisions(x, x.real + miny*1j, False) % 2 == 1

valid = lambda a, b: inside(interior(a, b, 1)[0][0]) and uniform(a, b)

def f2(li):
    return max(area_scaled(a, b)
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:])
               if valid(a, b)
               )


print("Part 1: ", int(f1(inpt)))
print("Part 2: ", int(f2(inpt)))

# wrong answers
# 1525207666
# 1525135584
# 1525063500