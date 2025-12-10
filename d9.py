## Time Stats:
# 0.92s user 0.05s system 92% cpu 1.041 total

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
# file = open("d9bb.txt")
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

def downsample(inpt, ratio=2):
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

# @cache
# def area_scaled(a, b):
#     return area(upsample(b) - upsample(a))

areas = sorted([(area_scaled(a,b), a, b)
                    for i, a in enumerate(inpt[:-1])
                    for _, b in enumerate(inpt[i + 1:])],
                key = lambda x: x[0])

def f1():
    return areas[-1][0]

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
    vbounds = set()
    hbounds = set()

    for a, b in bounds:
        if vert(a, b):
            vbounds.update(points(a, b))

        else:
            hbounds.update(points(a, b))

    return vbounds, hbounds

bounds = zip(inpt, inpt[1:] + inpt[:1])
vbounds, hbounds = splitbounds(bounds)

def flood(li, bounds):
# def flood(li):
    bounds = {bnd
              for a, b in zip(li, li[1:] + li[:1])
              for bnd in points(a, b)
              }    
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

outside = flood(inpt, vbounds | hbounds )
inside = lambda x: x not in outside

@cache
def collisions(a, b):
    if vert(a, b):
        dbounds = hbounds
    else:
        dbounds = vbounds

    return len(dbounds & set(points(a, b)))

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

uniform = lambda a, b: not any(collisions(na, nb) for na, nb in interior(a, b))

valid = lambda a, b: inside(interior(a, b)[0][0]) and uniform(a, b)

def f2():
    return next(filter(lambda x: valid(*x[1:]),
                       areas[::-1]))[0]

print("precompute finished!")
print("Part 1: ", int(f1()))
print("Part 2: ", int(f2()))

# wrong answers
# 1525207666
# 1525135584
# 1525063500