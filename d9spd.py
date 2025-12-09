## Time Stats:
# 1.41s user 0.06s system 95% cpu 1.540 total

# from collections import defaultdict as ddict, Counter as count
# from functools import reduce,  cmp_to_key, partial as par, cache
# from math import prod, sqrt as root, lcm as lcm, gcd as gcd
# from operator import itemgetter as ig
# from re import findall as rall

# import math
# import re

# from aoc import *

from functools import cache
import numpy as np

lmap = lambda f, li: [f(x) for x in li]
cp = lambda x, y: x + (y * 1j)

# file = open("d9.txt")
file = open("d9bb.txt")
raws = file.read().splitlines()
file.close()

rawss = """7,1
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
    return cp(a, b)

inpt = np.array(lmap(proc, raws))

def compute_areas(inpt):
    x = inpt.real.astype(np.int64) 
    y = inpt.imag.astype(np.int64)
    
    dx = np.abs(x[:, None] - x) + 1
    dy = np.abs(y[:, None] - y) + 1
    
    return dx * dy

areas = compute_areas(inpt)

def f1():
    return areas.max()

print("Part 1: ", int(f1()))

def downsample(inpt, ratio=2):
    x, y = inpt.real, inpt.imag
    
    xs = np.unique(x)
    ys = np.unique(y)
    
    xi = np.searchsorted(xs, x) * ratio
    yi = np.searchsorted(ys, y) * ratio
    
    downsampled = xi + yi * 1j
    
    idx_map = {complex(p): i for i, p in enumerate(downsampled)}
    
    def upsample(p):
        return idx_map[complex(p)]
    
    assert all(upsample(p) == i for i, p in enumerate(downsampled)), \
        "upsample doesn't preserve order"
    
    return downsampled, upsample

downsampled, upsample = downsample(inpt)
area_scaled = lambda a, b: areas[upsample(a), upsample(b)]

vert = lambda a, b: (b - a).real == 0

@cache
def points(a, b):
    if a == b:
        return (a,)

    d = b - a
    step = d/abs(d)

    return tuple(a + (x * step)
                 for x in range(int(abs(d)) + 1))

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

bounds = zip(inpt, inpt[1:] + inpt[:1])
vbounds, vbends, hbounds, hbends = splitbounds(bounds)

def flood(li, bounds):    
    mx = max(p.real for p in li)
    my = max(p.imag for p in li)

    valid = lambda x: -1 <= x.real <= mx + 1 and \
                      -1 <= x.imag <= my + 1
    
    done = set()
    curr = {-1 + -1j}

    while curr:
        curr = {np
                for p in curr
                for np in (p + (1j**k) for k in range(4))
                if np not in done 
                and np not in bounds
                and valid(np)}

        done |= curr
    
    return done

outside = flood(inpt, vbounds | vbends | hbounds | hbends)
inside = lambda x: x not in outside

print("precompute finished!")

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

inside = lambda x: collisions(x, x.real + miny*1j, False) % 2 == 1

valid = lambda a, b: inside(interior(a, b, 1)[0][0]) and uniform(a, b)

def f2(li):
    return max(area_scaled(a, b)
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:])
               if valid(a, b)
               )

# print("Part 2: ", int(f2(inpt)))

# BigBoy Answers
# p1: 275972310
# p2: 207548208