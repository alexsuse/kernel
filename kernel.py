"""
Solve the integral delayed equation

C(d) = K_0(d) - l/(a**2+C(0)) * \int_0^infinity C(s)C(s+d)ds

which gives the filtering kernel in the mean-field
approximation. The equation is solved by discretizing
C over a range from 0 to B, and assuming C = 0 for 
d>B. Then everything is done numerically. AWESOMES!
"""

import numpy
import matplotlib.pyplot as plt

def SquaredDistance(f,g):
    """
    Squared distance between two functions
    discretized as arrays.
    """
    assert f.shape == g.shape
    dist = 0.0
    for i,j in zip(f,g):
        dist += (i-j)**2
    return dist

def IterateOnce(C, K0, dx, la = 1.0, alpha=1.0):
    """
    Iterate the integral equation once
    """
    newC = numpy.zeros_like(C)
    for i in range(newC.size):
        Cdelta = numpy.roll(C,i)
        Cdelta[:i] = 0.0
        newC[i] -= la*numpy.sum(C*Cdelta)*dx/(alpha**2+C[0])
        newC[i] += K0(i*dx)
    return newC

if __name__=="__main__":
    k = 2.0
    K = lambda x : (1.0+k*numpy.abs(x))*numpy.exp(-k*numpy.abs(x))/(4*k**3)
    K0 = numpy.vectorize(K)
    dx = 0.01
    xs = numpy.arange(0.0,10000*dx,dx)
    C0 = K0(xs)
    C1 = IterateOnce(C0, K0, dx,alpha=0.1)
    dist = SquaredDistance(C0,C1)
    while dist > 1e-10:
        C0 = C1
        C1 = IterateOnce(C0, K0, dx,alpha =0.1)
        dist = SquaredDistance(C0,C1)
        print dist

    plt.plot(xs,(K0(xs)),xs,(C1))
    plt.show()

