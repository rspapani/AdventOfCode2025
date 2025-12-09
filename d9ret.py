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

sraws = """7,1
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

def f1(li):
    return max(area_scaled(a,b)
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:]))

def points(a, b):
    d = b - a
    step = d/abs(d)

    return [a + (x * step)
            for x in range(int(abs(d)) + 1)]

def flood(li):
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

outside = flood(inpt)

def covered(a, b):
    x1, x2 = sorted((int(a.real), int(b.real)))
    y1, y2 = sorted((int(a.imag), int(b.imag)))
    return not any(x + y*1j in outside 
                   for x in range(x1, x2) 
                   for y in range(y1, y2))

def f2(li):
    return max((int(area_scaled(a, b)), f"{upsample(a)}, {upsample(b)}")
               for i, a in enumerate(li[:-1])
               for _, b in enumerate(li[i + 1:])
               if covered(a, b)
               )


print("Part 1: ", int(f1(inpt)))
print("Part 2: ", f2(inpt)[0])

# wrong answers
# 1525207666
# 1525135584
# 1525063500