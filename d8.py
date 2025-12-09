from collections import defaultdict as ddict, Counter as count
from functools import reduce,  cmp_to_key, partial as par
from math import prod, sqrt as root, lcm as lcm, gcd as gcd
from operator import itemgetter as ig
from re import findall as rall

import math
import re

from aoc import *

file = open("d8.txt")
raws = file.read().splitlines()
file.close()

rasws = """162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689""".splitlines()

def proc(x):
    return vec(map(int, x.split(',')))

inpt = lmap(proc, raws)

dist = lambda a, b: (a - b).norm()
getdists = lambda li: sorted([(dist(x, y), i, i + 1 + j) 
                                for i, x in enumerate(li)
                                for j, y in enumerate(li[i + 1:])])

def find_subgraph(x, cons):
    done = set()
    curr = {x}

    while curr:
        curr = {j 
                for i in curr
                for j in cons[i]
                if j not in done}

        done |= curr

    return done

def f1(li, n = 1000):
    dists = getdists(li)

    cons = ddict(list)

    for _, i, j in dists[:n]:
        cons[i].append(j)
        cons[j].append(i)

    circs = []
    seen = set()

    for i in cons:
        if i not in seen:
            circ = find_subgraph(i, cons)
            circs.append(len(circ))
            seen |= circ


    return prod(sorted(circs)[-3:])

def get_graph(x, graphs):
    return next(filter(
                        lambda g: x in g[1],
                        enumerate(graphs)
                        ),
                (-1, 0)
                )[0]

def f2(li):
    dists = getdists(li)
    seen = 0
    graphs = []

    # print("\n".join(map(str, enumerate(li))))
    # print("\n".join(map(str, dists)))

    for _, i, j in dists:
        # print(graphs, i, j)
        gi = get_graph(i, graphs)
        gj = get_graph(j, graphs)

        seen += (gi == -1) + (gj == -1)

        if gi == -1 and gj == -1:
            graphs.append({i, j})

        elif gi == -1:
            graphs[gj].add(i)

        elif gj == -1:
            graphs[gi].add(j)

        elif gi != gj:
            mn, mx = sorted((gi, gj))
            graphs[mn] |= graphs[mx]
            graphs.pop(mx)

        if seen == len(li) and len(graphs) == 1:
            return li[i][0] * li[j][0]


print("Part 1: ", f1(inpt))
print("Part 2: ", f2(inpt))
