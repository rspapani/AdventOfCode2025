from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par, cache
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d12.txt")
raws = file.read()
file.close()

rawss = """0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
"""

raws = raws.split('\n\n')

def proc(x):
    dims, li = x.split(': ')
    dx, dy = lmap(int, dims.split('x'))
    return (dx, dy), lmap(int, li.split())

def loadshape(shape):
    outs = []
    for  y, row in enumerate(shape):
        for x, ch in enumerate(row):
            if ch == '#':
                nx, ny = x - 1, y - 1
                outs.append(complex(nx, ny))

    return tuple(outs)

shapes = [loadshape(chunk.splitlines()[1:])
          for chunk in raws[:-1]]
inpt = lmap(proc, raws[-1].splitlines())

print(shapes, inpt)

@cache
def rotate(shape):
    return tuple(k * (1j) for k in shape)

@cache
def translate(pos, shape):
    shift = complex(*pos) + complex(1, 1)
    return tuple(k + shift for k in shape)

def canfit(dims, reqs):
    print(dims)
    mx, my = dims
    space = [[0 for _ in range(mx)] 
             for _ in range(my)]

    def get(k):
        kx, ky = int(k.real), int(k.imag)
        return space[ky][kx]

    def put(k, val = 1):
        kx, ky = int(k.real), int(k.imag)
        space[ky][kx] = val

    def place(shape):
        for p in shape:
            put(p, 1)

    def remove(shape):
        for p in shape:
            put(p, 0)

    valid = lambda k: 0 <= k.real < mx and \
                      0 <= k.imag < my and \
                      get(k) == 0
    
    canplace = lambda shape: all(valid(p) for p in shape)
    coords = [(x, y) for x in range(mx) 
                     for y in range(my)]

    n = len(reqs)
    steps = 0
    def step(i):
        nonlocal steps
        # print(steps, i)
        if i == n:
            return True
        elif reqs[i] == 0:
            return step(i + 1)
        
        curr = shapes[i]

        for r in range(4):
            curr = rotate(curr)
            for pos in coords:

                if get(complex(*pos)) == 0:

                    cand = translate(pos, curr)
                    if canplace(cand):
                        place(cand)
                        reqs[i] -= 1

                        if step(i):
                            return True
                        
                        remove(cand)
                        reqs[i] += 1
                
        return False  

    return step(0)

@cache
def area(i):
    # return 9
    return len(shapes[i])

for i, _ in enumerate(shapes):
    print(i, area(i))

def hasspace(dim, reqs):
    available = prod(dim)
    needed = sum(cnt*area(i) 
                 for i, cnt in enumerate(reqs))
    
    print(available, needed)
    return available >= needed

def f1(li):
    return sum(hasspace(dim, reqs)
               for dim, reqs in li)

def f2(li):
    pass

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
