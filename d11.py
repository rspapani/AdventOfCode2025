from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par, cache
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d11.txt")
raws = file.read().splitlines()
file.close()

rawss = """aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out
""".splitlines()

rawss = """svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out
""".splitlines()

def proc(x):
    graph = ddict(list)

    for row in x:
        curr, paths = row.split(': ')
        for path in paths.split():
            graph[curr].append(path)

    return graph

inpt = proc(raws)

def f1(li, strt = "you", goal = "out"):
    @cache
    def nopaths(curr):        
        return 1 if curr == goal \
                 else sum(nopaths(path) 
                          for path in li[curr])
    
    return nopaths(strt)

def f2(li):
    paths = [["svr", "fft", "dac", "out"],
             ["svr", "dac", "fft", "out"]]

    return sum(prod(f1(li, strt=a, goal=b)
                    for a,b in zip(path, path[1:]))
                for path in paths)

print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
