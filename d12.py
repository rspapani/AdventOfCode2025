#An actual solution to day 12, solves the actual problem instead of copping out
#runtime: 2235.57s user 55.69s system 159% cpu 23:58.30 total

from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par, cache
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

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

from ortools.sat.python import cp_model
from aoc import *

file = open("d12.txt")
raws = file.read()
file.close()

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

rotate = lambda shape, n: tuple(k * ((1j)**n) for k in shape)
allrotations = lambda shape: [rotate(shape, i) for i in range(4)]

ctot = lambda p: (int(p.real), int(p.imag))

def dedup(shapes):
    outs = []

    for shape in shapes:
        curr = set(shape)
        if not any(all(p in curr
                       for p in pv)
                   for pv in outs):
            outs.append(shape)

    return outs

rotations = [dedup(allrotations(shape))
             for _, shape in enumerate(shapes)]

@cache
def translate(pos, shape):
    shift = complex(*pos) + complex(1, 1)
    return tuple(k + shift for k in shape)

areas = [len(shape)
         for _, shape in enumerate(shapes)]

def hasspace(dim, reqs):
    available = prod(dim)
    needed = sum(cnt*areas[i] 
                 for i, cnt in enumerate(reqs))
    
    return available >= needed

@cache
def get_squares(x, y, si, ri):
    shape = rotations[si][ri]
    return translate((x, y), shape)

inpt = lmap(proc, raws[-1].splitlines())

print("PRECOMPUTE FINISHED")

def feasible(dim, reqs):
    print(f"MODELING {dim} {reqs}")

    mx, my = dim

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    model = cp_model.CpModel()

    bools = {}
    collisions = [[] for _ in range(mx * my)]
    counts = [[] for _ in shapes]

    ptoi = lambda p: int((p.imag * mx) + p.real)

    for x in range(mx - 2):
        for y in range(my - 2):
            for si, _ in enumerate(shapes):
                for ri, _ in enumerate(rotations[si]):
                    key = (x, y, si, ri)
                    
                    bools[key] = model.NewBoolVar(f"{x}_{y}_{si}_{ri}")

                    counts[si].append(bools[key])
                    for square in get_squares(*key):
                        collisions[ptoi(square)].append(bools[key])

    # print(f"VECTOR DEFINED WITH {len(bools)} VARS")
    for coll in collisions:
        model.Add(sum(coll) <= 1)

    for i, count in enumerate(counts):
        model.Add(sum(count) == reqs[i])

    # print(f"CONSTRAINTS DEFINED: {len(counts) + len(collisions)}")

    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        print("FEASIBLE")
        return True
    elif status == cp_model.INFEASIBLE:
        print("INFEASIBLE")
        return False
    else:
        print(f"TIMEOUT ON {dim} {reqs}")
        return False

def f1(li):
    return sum(feasible(dim, reqs)
               for dim, reqs in li
               if hasspace(dim, reqs))

print("Part 1: ", f1(inpt))

def f2(li):
    return ":)"
print("Part 2: ", f2(inpt))
