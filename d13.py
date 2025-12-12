# from collections import defaultdict as ddict, Counter as count
# from functools import reduce,  cmp_to_key, partial as par
# from math import prod, sqrt as root, lcm as lcm, gcd as gcd
# from operator import itemgetter as ig
# from re import findall as rall

# import math
# import re

# from aoc import *
from aoc import lmap, vec, adjs

file = open("d13beta.txt")
raws = file.read()
file.close()

raws = raws.split('\n\n')

def proc(slices):
    blocks = set()
    holes = set()

    mz, mx, my = len(slices), len(slices[0]), len(slices[0][0])

    for z, slice in enumerate(slices):
        for y, row in enumerate(slice):
            for x, ch in enumerate(row):
                if ch == '#':
                    blocks.add(vec((x, y, z)))
                elif z == 0:
                    holes.add(vec((x, y, z)))

    return blocks, holes, vec((mx, my, mz))

inpt = proc(lmap(lambda x: x.splitlines(), raws))

def f1(li):
    blocks, curr, ceil = li

    floor = vec((0, 0, 0))
    valid = lambda p: floor <= p < ceil

    visited = set()
    visited |= curr

    while curr:
        stepped = {npos
                   for pos in curr
                   for npos in adjs(pos)
                   if valid(npos)
                   and npos not in visited}

        visited |= stepped
        curr = stepped - blocks

    return len(visited & blocks), visited

def f2(li):
    _, exposed = f1(li)
    blocks, _, ceil = li 

    invalid = exposed | blocks

    mx, my, mz = ceil

    return sum((x, y, z) not in invalid
               for x in range(mx)
               for y in range(my)
               for z in range(mz))

print("Part 1: ", f1(inpt)[0])
print("Part 2: ", f2(inpt))
