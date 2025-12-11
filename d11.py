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

i = 0
def proc2(x):
    graph = ddict(list)
    mapping = {}

    def getind(key):
        global i
        if key not in mapping:
            mapping[key] = i
            i += 1

        return mapping[key]

    for row in x:
        curr, paths = row.split(': ')
        ci = getind(curr)
        for path in paths.split():
            pi = getind(path)
            graph[ci].append(pi)

    return [graph[ni] for ni in range(i)], mapping


inpt = proc2(raws)
# print(inpt)


def f1(li, sstrt = "you", sgoal = "out", needs = []):
    li, mapping = li

    strt = mapping[sstrt]
    goal = mapping[sgoal]

    press = []

    for need in needs:
        press.append(mapping[need])

    @cache
    def nopaths(curr, visited = 0):
        # print(curr, visited)
        invisited = lambda k: bool(visited & (1<<k))
        addvisit = lambda k: visited & (1<<k)

        if curr == goal:
            return all(invisited(k) for k in press)
        
        nvisited = addvisit(curr)
        res = sum(nopaths(path, nvisited)
                  for path in li[curr]
                  if not invisited(path))

        return res
    
    return nopaths(strt)

def f12(li, sstrt = "you", sgoal = "out"):
    li, mapping = li

    strt = mapping[sstrt]
    goal = mapping[sgoal]

    @cache
    def nopaths(curr):
        if curr == goal:
            return 1
        
        res = sum(nopaths(path)
                  for path in li[curr])

        return res
    
    return nopaths(strt)

def f2(li):
    
    paths = [["svr", "fft", "dac", "out"],
             ["svr", "dac", "fft", "out"]]
    
    outs = 0
    for path in paths:
        res = 1

        for a,b in zip(path, path[1:]):
            res *= f12(li, sstrt=a, sgoal=b)

        outs += res

    return outs

print("Part 1: ", f1(inpt))
print("Part 1 v2: ", f12(inpt))
# print("Part 2: ", f12(inpt, sstrt="svr"))
print("Part 2: ", f2(inpt))
