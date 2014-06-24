"""
Solve the integral delayed equation

C(d) = K_0(d) - l/(a**2+C(0)) * \int_0^infinity C(s)C(s+d)ds

which gives the filtering kernel in the mean-field
approximation. The equation is solved by discretizing
C over a range from 0 to B, and assuming C = 0 for 
d>B. Then everything is done numerically. AWESOMES!
"""

import numpy
import prettyplotlib as ppl
from prettyplotlib import plt

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

def MakeGram(t,K):
    if not t:
        return numpy.array([[0]])
    ts = numpy.tile(t,(len(t),1))
    tsl = numpy.tile(numpy.array(t).reshape((len(t),1)),(1,len(t)))
    return K(ts-tsl)

def MakeVector(t,K):
    return K(t)

def GetStochasticEps(params):
    alpha, la, dt, N = params
    K = lambda x : numpy.exp(-2.0*numpy.array(x)**2)
    maxtimes = 60
    eps = numpy.zeros(N)
    eps[:] = K(0)
    times = []
    while len(times) < maxtimes:
        times = map(lambda x: x+dt, times)
        if numpy.random.rand() < la*dt:
            times.append(0)
            if len(times) > maxtimes:
                times = times[1:]

    for i in range(N):
        times = map(lambda x: x+dt, times)
        if numpy.random.rand() < la*dt:
            times.append(0)
            if len(times) > maxtimes:
                times = times[1:]
        if times:
            G = MakeGram(times,K)
            C = G + alpha**2*numpy.eye(G.shape[0])
            eps[i] = K(0) - numpy.dot(K(times),numpy.linalg.solve(G,K(times)))
    return numpy.mean(eps)
    
    

if __name__=="__main__":
    k = 2.0
    K_rbf = lambda x : numpy.exp(-k*numpy.array(x)**2)
    print GetStochasticEps(K_rbf, 0.1,1.0,0.01,10000)
    K_matern = lambda x : (1.0+k*numpy.abs(x))*numpy.exp(-k*numpy.abs(x))
    K_ou = lambda x : numpy.exp(-k*numpy.abs(x))
    dx = 0.0005
    xs = numpy.arange(0.0,6000*dx,dx)
    rbf0 = K_rbf(xs)
    rbf1 = IterateOnce(rbf0, K_rbf, dx,alpha=0.1)
    dist = SquaredDistance(rbf0,rbf1)
    while dist > 1e-10:
        rbf0 = rbf1
        rbf1 = IterateOnce(rbf0, K_rbf, dx,alpha =0.1)
        dist = SquaredDistance(rbf0,rbf1)
        print dist

    matern0 = K_matern(xs)
    matern1 = IterateOnce(matern0, K_matern, dx,alpha=0.1)
    dist = SquaredDistance(matern0,matern1)
    while dist > 1e-10:
        matern0 = matern1
        matern1 = IterateOnce(matern0, K_matern, dx,alpha =0.1)
        dist = SquaredDistance(matern0,matern1)
        print dist

    OU0 = K_ou(xs)
    OU1 = IterateOnce(OU0, K_ou, dx,alpha=0.1)
    dist = SquaredDistance(OU0,OU1)
    while dist > 1e-10:
        OU0 = OU1
        OU1 = IterateOnce(OU0, K_ou, dx,alpha =0.1)
        dist = SquaredDistance(OU0,OU1)
        print dist

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    plt.gcf().suptitle("Posterior kernels")

    ax1.plot(xs,K_ou(xs),label=r'OU prior kernel')
    ax1.plot(xs,OU1,label = r'OU posterior kernel')
    ax1.legend()
    ax2.plot(xs,K_matern(xs),label=r'Matern prior kernel')
    ax2.plot(xs,matern1, label=r'Matern posterior kernel')
    ax2.legend()
    ax3.plot(xs,K_rbf(xs),label=r'RBF prior kernel')
    ax3.plot(xs,rbf1,label=r'RBF posterior kernel')
    ax3.legend()

    plt.savefig("figure_3_3.eps")
