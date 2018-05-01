#!/usr/bin/env python3
"""Draw dynamical Lakes of Wada on the torus (method of Coudene)"""
import time
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-n","--size",help="size in pixels of output image",type=int,default=100)
parser.add_argument("-a","--amplitude",help="intensity of perturbation of the Anosov diffeomorphism",type=float,default=2.2)
parser.add_argument("-s","--spread",help="size of region in which to perturb the Anosov diffeomorphism",type=float,default=0.23)
parser.add_argument("-i","--islands",help="number of islands (must be 1 or 4)",type=int,default=4)
parser.add_argument("-c","--colormap", help="output colormap to use (anything matplotlib recognizes)",default="plasma")
parser.add_argument("-o","--output", help="output filename (default is time- and parameter-based)")
args = parser.parse_args()

if args.islands not in [1,4]:
    print("Invalid number of islands ({}).  Must be 1 or 4.".format(args.islands))
    sys.exit(1)

phi = 1.6180339887498948482
A = np.array( [[2,1],[1,1]] )
B = (1.0 / (1 + phi*phi)) * np.array( [[phi*phi,phi],[phi,1]] )

def wrap(p):
    return ((p + np.array([0.5,0.5])) % np.array([1,1])) - np.array([0.5,0.5])

def k(t):
    if abs(t) > 1:
        return 0.0
    else:
        return (1-t*t)**2

def ctr(p):
    if args.islands == 1:
        return p - np.floor(p + 0.5)
    else:
        return p - 0.5*np.floor(2.0*p + 0.5)

def f(p):
    p2 = ctr(p)
    x2,y2 = p2
    return wrap( A.dot(p) - args.amplitude*k(x2/args.spread)*k(y2/args.spread)*(B.dot(p2)) )

eps = 0.0001

def nearfix1(p):
    x,y = p
    if abs(x) < eps and abs(y) < eps:
        return 1
    else:
        return 0

def nearfix4(p):
    x,y = p
    if abs(x) < eps:
        if abs(y) < eps:
            return 1
        elif abs(abs(y) - 0.5) < eps:
            return 2
        else:
            return 0
    elif abs(abs(x) - 0.5) < eps:
        if abs(y) < eps:
            return 3
        elif abs(abs(y) - 0.5) < eps:
            return 4
        else:
            return 0
    else:
        return 0

def pointcolor(p0):
    p = p0
    N=0
    near = 0
    while not near:
        if N>500:
            break
        p = f(p)
        N = N+1
        if args.islands == 1:
            near = nearfix1(p)
        else:
            near = nearfix4(p)
    if args.islands == 1:
        x,y = p
        if x == 0.0 and y == 0.0:
            return 0
        q = x / np.sqrt(x*x + y*y)
        return 0.5*(1+q)
        
    if near <= 1:
        return near
    else:
        return (near + N) % 3 + 2

nx=args.size
ny=args.size
c = np.zeros((nx,ny))
for j in range(ny):
    print('row {} of {}'.format(j+1,args.size))
    for i in range(nx):
        p = np.array( [-0.5+ (i/float(nx-1)), 0.5 - (j/float(ny-1))] )
        c[i,j] = pointcolor(p)
#        print(c[i,j])

if not args.output:
    args.output = 'wada-torus-i{}-n{}-{}.png'.format(args.islands,args.size,int(time.time()))
    print("Using output filename: {}".format(args.output))
                                                 
plt.imsave(args.output, c, cmap=args.colormap)

