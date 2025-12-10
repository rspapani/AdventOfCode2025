## Time Stats:
# Standard
# 0.17s user 0.04s system 53% cpu 0.404 total

# BigBoy
# 56.97s user 23.48s system 90% cpu 1:29.27 total

# BigBoy Answers
# p1: 275972310
# p2: 207548208

import numpy as np
from scipy import ndimage

# if u go higher it will probably crash ur computer
BATCH_SIZE = 10**7

lmap = lambda f, li: [f(x) for x in li]

file = open("d9.txt")
# file = open("d9.bb")
raws = file.read().splitlines()
file.close()

def proc(x):
    a,b = lmap(int, x.split(','))
    return complex(a, b)

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

## Pre Computes
# Downsampling
def downsample(inpt, ratio=2):
    x, y = inpt.real, inpt.imag
    
    xs = np.unique(x)
    ys = np.unique(y)
    
    xi = np.searchsorted(xs, x) * ratio
    yi = np.searchsorted(ys, y) * ratio
    
    downsampled = xi + yi * 1j
    
    return downsampled

downsampled = downsample(inpt)
vert = lambda a, b: (b - a).real == 0

# Boundaries
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

    return np.array(list(vbounds)), np.array(list(hbounds))

bounds = zip(downsampled, np.roll(downsampled, -1))
vbounds, hbounds = splitbounds(bounds)

# Floodfill
def flood(li, vbounds, hbounds):
    mx = int(max(p.real for p in li))
    my = int(max(p.imag for p in li))
    
    w, h = mx + 3, my + 3
    
    gaps = np.ones((w, h), dtype=bool)
    
    bounds = np.concatenate([vbounds, hbounds])
    bx = (bounds.real + 1).astype(int)
    by = (bounds.imag + 1).astype(int)
    gaps[bx, by] = False
    
    labeled, _ = ndimage.label(gaps)
    
    return labeled == labeled[0, 0]

outside = flood(downsampled, vbounds, hbounds)
inside_grid = ~outside[1:-1, 1:-1]

prefix = np.cumsum(np.cumsum(inside_grid.astype(np.int32), axis=0), axis=1)
prefix = np.pad(prefix, ((1, 0), (1, 0)), mode='constant', constant_values=0)

print("precompute finished!")

#why yes I have heard of overparametrizing
def batch_covered(batch_indices, n, x, y, prefix, areas):
    i, j = np.divmod(batch_indices, n)
    mask = i < j
    i, j = i[mask], j[mask]
    
    mnx = np.minimum(x[i], x[j])
    mxx = np.maximum(x[i], x[j])
    mny = np.minimum(y[i], y[j])
    mxy = np.maximum(y[i], y[j])
    
    present = prefix[mxx+1, mxy+1] + prefix[mnx, mny]\
            - (prefix[mnx, mxy+1] + prefix[mxx+1, mny]) 
    
    needed = (mxx - mnx + 1) * (mxy - mny + 1)
    
    valid = present == needed
    if valid.any():
        first_valid = np.argmax(valid)
        return areas[i[first_valid], j[first_valid]]
    
    return False

def f2():
    batch_size = BATCH_SIZE
    n = len(downsampled)
    x = downsampled.real.astype(np.int32)
    y = downsampled.imag.astype(np.int32)
    
    areaidxs = np.argsort(areas[:n, :n].ravel())[::-1]
    
    for start in range(0, len(areaidxs), batch_size):
        res = batch_covered(areaidxs[start:start + batch_size], n, x, y, prefix, areas)
        if res:
            return res
        
    return 0 #Critical Fault

print("Part 2: ", int(f2()))
