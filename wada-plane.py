#!/usr/bin/sage -python
"""Draw dynamical Lakes of Wada on the plane (method of Coudene; project with Weierstrass P function)"""

# This script uses Sagemath for the sole purpose of access to the
# special function "elliptic_f" which allows us to inverrt the
# Weierstrass P function.  It would be a good idea to substitute an
# implementation of this one function so that the script could run on
# stock Python3 with only numpy and matplotlib.

# NOTE: You may need to alter the path to the sage binary in the first
# line of the file, or run it with
#
#   /path/to/sage -python wada-plane.py
#
# We don't use /usr/bin/env to locate the sage binary because most
# posix-like systems only allow one parameter to the file interpreter
# (preventing us from passing -python).

import sage.all
from sage.functions.special import elliptic_f
from sage.functions.trig import arcsin, arctan
from matplotlib import pyplot as plt
import time
import numpy as np
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-n","--size",help="size in pixels of output image",type=int,default=100)
parser.add_argument("--amplitude1", help="intensity of perturbation of the Anosov diffeomorphism at the origin",type=float,default=2.4)
parser.add_argument("--spread1", help="size of region in which to perturb the Anosov diffeomorphism at the origin",type=float,default=0.5)
parser.add_argument("--amplitude2", help="intensity of perturbation of the Anosov diffeomorphism at the 2-torsion points",type=float,default=2.3)
parser.add_argument("--spread2", help="size of region in which to perturb the Anosov diffeomorphism at the 2-torsion points",type=float,default=0.25)
parser.add_argument("-r","--radius", help="size of the window in the complex plane to draw",type=float,default=4)
parser.add_argument("-t","--theta", help="CW rotation of image (radians)",type=float,default=1.7)
parser.add_argument("-c","--colormap", help="output colormap to use (anything matplotlib recognizes)",default="magma")
parser.add_argument("-o","--output", help="output filename (default is time- and parameter-based)")
args = parser.parse_args()

phi = 1.6180339887498948482
A = np.array( [[2,1],[1,1]] )
B = (1.0 / (1 + phi*phi)) * np.array( [[phi*phi,phi],[phi,1]] )

def wrap(p):
    return ((p + np.array([0.5,0.5])) % np.array([1,1])) - np.array([0.5,0.5])

def k(x,y):
    r2 = x*x + y*y
    if r2 > 1:
        return 0.0
    return (1-r2)**2

def ctr(p):
    return p - 0.5*np.floor(2.0*p + 0.5)

def f(p):
    x,y = p
    if abs(x) > 0.25 or abs(y) > 0.25:
        p2 = ctr(p)
        x2,y2 = p2
        v2 = k(x2/args.spread2,y2/args.spread2)*(B.dot(p2))
    else:
        v2 = np.array([0,0])
    v1 = A.dot(p) - args.amplitude1*k(x/args.spread1,y/args.spread1)*B.dot(p) - args.amplitude2*v2
    return wrap(v1)

eps = 0.0001

def nearfix(p):
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

def itercount(p0):
    p = p0
    N=0
    near = 0
    while not near:
        if N>500:
            break
        p = f(p)
        N = N+1
        near = nearfix(p)
    if near <= 1:
        return near
    else:
        return (near + N) % 3 + 2

invper = 0.38137988175090659403 # Gamma(3/4) / (2 sqrt(pi) Gamma(5/4) )
    
def invP(z):
    t = complex(z)**(-0.5)
    phi = arcsin(t)
    return invper*elliptic_f(phi,-1)

nx=args.size
ny=args.size

c = np.zeros((nx,ny))
for j in range(ny):
    print('row {} of {}'.format(j+1,args.size))
    for i in range(nx):
        z = args.radius*np.exp(args.theta*1j)*((-1 + 2*(i/float(nx-1))) + 1j*(-1 + 2*(j/float(ny-1))))
        
        if z != 0:
            w = invP(z)
            c[i,j] = itercount(np.array([w.real,w.imag]))
        else:
            c[i,j] = 1

if not args.output:
    args.output = 'wada-plane-n{}-r{}-t{}-{}.png'.format(args.size,args.radius,args.theta,int(time.time()))
    print("Using output filename: {}".format(args.output))
                                                 
plt.imsave(args.output, c, cmap=args.colormap)
