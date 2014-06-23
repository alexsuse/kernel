"""
Solve the integral delayed equation

C(d) = K_0(d) - l/(a**2+C(0)) * \int_0^infinity C(s)C(s+d)ds

which gives the filtering kernel in the mean-field
approximation. The equation is solved by discretizing
C over a range from 0 to B, and assuming C = 0 for 
d>B. Then everything is done numerically. AWESOMES!
"""

import numpy
import cPickle as pic
from kernel import SquaredDistance, IterateOnce

def get_eq_RBF_eps(K_rbf, xs, dx, alpha, la):

    rbf0 = K_rbf(xs)
    rbf1 = IterateOnce(rbf0, K_rbf, dx,alpha=alpha, la=la)
    dist = SquaredDistance(rbf0,rbf1)
    while dist > 1e-8:
        rbf0 = rbf1
        rbf1 = IterateOnce(rbf0, K_rbf, dx,alpha =alpha, la=la)
        dist = SquaredDistance(rbf0,rbf1)

    return rbf1[0]

if __name__=="__main__":
    k = 2.0
    K_rbf = lambda x : numpy.exp(-k*x**2)
    dx = 0.0005
    xs = numpy.arange(0.0,6000*dx,dx)
    
    dalpha = 0.1
    dphi = 0.1

    alphas = numpy.arange(0.01,3.0,dalpha)
    phis = numpy.arange(0.01,2.0,dalpha)

    eps = numpy.zeros((alphas.size,phis.size))

    try:
        fi = open("rbf_eps.pik","rb")
        print "Found pickle, skipping simulation"
        eps = pic.load(fi)

    except:
        print "No pickle, rerunning"

        for i,a in enumerate(alphas):
            for j,p in enumerate(phis):
                print i,j
                la = numpy.sqrt(2*numpy.pi)*a*p
                eps[i,j] = get_eq_RBF_eps(K_rbf, xs, dx, a, la)

        with open("rbf_eps.pik","wb") as fi:
            pic.dump(eps,fi)

    import prettyplotlib as ppl
    from prettyplotlib import plt
    from prettyplotlib import brewer2mpl

    fig, (ax1,ax2) = ppl.subplots(2,figsize=(20,8))


    alphas2,phis2 = np.meshgrid(np.arange(alphas.min(),alphas.max()+dalpha,dalpha)\
                                               -dalpha/2,
                                np.arange(phis.min(),phis.max()+dphi,dphi)-dphi/2)
    yellorred = brewer2mpl.get_map('YlOrRd','Sequential',9).mpl_colormap

    p = ax1.pcolormesh(alphas2,phis2,eps.T,cmap=yellorred)
    ax1.axis([alphas2.min(),alphas2.max(),phis2.min(),phis2.max()])

    ax1.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\phi$')

    ppl.plot(alphas, eps[:,1], label=r'$\phi = '+str(phis[1])+r'$',ax=ax2)
    ppl.plot(alphas, eps[:,10], label=r'$\phi = '+str(phis[10])+r'$',ax=ax2)
    ppl.plot(alphas, eps[:,-1], label=r'$\phi = '+str(phis[-1])+r'$',ax=ax2)
    
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\epsilon$')

    plt.savefig("figure_5_7.eps")
