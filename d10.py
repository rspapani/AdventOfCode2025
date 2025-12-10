from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par, cache
from itertools import zip_longest
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

#fuck you eric
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

import math
import re

from aoc import *

file = open("d10.txt")
raws = file.read().splitlines()
file.close()

sraws = """[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
""".splitlines()

@cache
def ttob(tup):
    return reduce(lambda a, b: a ^ (1<<b),
                  tup, 0)

def proc(x):
    line = x.split()
    
    diag = tuple((i for i, ch 
                    in enumerate(line[0][1:-1])
                    if ch =='#'))
    
    tobut = lambda b: ttob(map(int, 
                               b[1:-1].split(',')
                            ))
    butts = line[1:-1]

    jolts = map(int, line[-1][1:-1].split(','))

    return ttob(diag), lmap(tobut, butts), tuple(jolts)

inpt = lmap(proc, raws)

def dfs(targ, paths):
    if not targ:
        return 0
    
    elif not paths:
        return math.inf
    
    return min(dfs(targ ^ paths[0], paths[1:]) + 1,
               dfs(targ, paths[1:]))

def f1(li):
    return sum(dfs(targ, paths)
               for targ, paths, _ in li)

def b2v(n, b):
    pres = lambda x: int(bool(b & (1<<x)))
    return np.array([pres(i) for i in range(n)])

def bs2a(n, bs):
    return np.array([b2v(n, b) for b in bs]).T

def solveILP(A, b, nx):
    # print(A, b, n)
    return milp( c=np.ones(nx),  
                 constraints=LinearConstraint(A, b, b),
                 integrality=np.ones(nx), 
                 bounds=Bounds(lb=0, ub=np.inf)
                )

def solverow(targ, pths):
    a = bs2a(len(targ), pths)
    b = np.array(targ)
    nx = len(pths)

    res = solveILP(a, b, nx)
    return int(res.x.sum())

def f2(li):
    return sum(solverow(targ, paths)
               for _, paths, targ in li)

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
