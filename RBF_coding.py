"""
Solve the integral delayed equation

C(d) = K_0(d) - l/(a**2+C(0)) * \int_0^infinity C(s)C(s+d)ds

which gives the filtering kernel in the mean-field
approximation. The equation is solved by discretizing
C over a range from 0 to B, and assuming C = 0 for 
d>B. Then everything is done numerically. AWESOMES!
"""

import sys
import numpy
import cPickle as pic
from kernel import SquaredDistance, IterateOnce, GetStochasticEps
import multiprocessing as mp

def get_eq_RBF_eps(params):
    xs, dx, alpha, la = params

    K_rbf = lambda x : numpy.exp(-2.0*numpy.array(x)**2)
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
    K_rbf = lambda x : numpy.exp(-2.0*numpy.array(x)**2)
    dx = 0.0005
    xs = numpy.arange(0.0,6000*dx,dx)
    
    dalpha = 0.1
    dphi = 0.1

    alphas = numpy.arange(0.01,3.0,dalpha)
    phis = numpy.arange(0.1,2.01,dphi)

    eps = numpy.zeros((alphas.size,phis.size))

    try:
        sys.argv[1]
        finame = sys.argv[1]
    except:
        finame = "rbf_eps_stoc.pik"

    try:
        fi = open(finame,"rb")
        print "Found pickle, skipping simulation"
        eps,stoc_eps_02,stoc_eps_10,stoc_eps_20 = pic.load(fi)
        print 'all good with the pickle'

    except:
        print "No pickle, rerunning"
        
        ncpus = mp.cpu_count()
        pool = mp.Pool(processes=ncpus)

        print "Running on a pool of "+str(ncpus)+" cpus"
        
        params02 = [(a,numpy.sqrt(2*numpy.pi)*a*phis[1],0.0001,400000) for a in alphas]
        params10 = [(a,numpy.sqrt(2*numpy.pi)*a*phis[9],0.0001,400000) for a in alphas]
        params20 = [(a,numpy.sqrt(2*numpy.pi)*a*phis[-1],0.0001,400000) for a in alphas]
        stoc_eps_02 = numpy.array(pool.map(GetStochasticEps,params02))
        stoc_eps_10 = numpy.array(pool.map(GetStochasticEps,params10))
        stoc_eps_20 = numpy.array(pool.map(GetStochasticEps,params20))

        for i,a in enumerate(alphas):
            print i
            params = [(xs,dx,a,numpy.sqrt(2*numpy.pi)*a*p) for p in phis]
            eps[i,:] = numpy.array(pool.map(get_eq_RBF_eps,params))


        with open(finame,"wb") as fi:
            pic.dump([eps,stoc_eps_02,stoc_eps_10,stoc_eps_20],fi)
            print "dumped the pickle"

    import prettyplotlib as ppl
    from prettyplotlib import plt
    from prettyplotlib import brewer2mpl

    print 'imported'

    font = {'size':16}
    plt.rc('font',**font)

    fig, (ax1,ax2) = ppl.subplots(1,2,figsize=(20,8))


    alphas2,phis2 = numpy.meshgrid(numpy.arange(alphas.min(),alphas.max()+dalpha,dalpha)\
                                               -dalpha/2,
                                numpy.arange(phis.min(),phis.max()+dphi,dphi)-dphi/2)

    print 'mesh is good'

    yellorred = brewer2mpl.get_map('YlOrRd','Sequential',9).mpl_colormap

    p = ax1.pcolormesh(alphas2,phis2,eps.T,cmap=yellorred)
    ax1.axis([alphas2.min(),alphas2.max(),phis2.min(),phis2.max()])

    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\phi$')

    ax1.set_title(r'MMSE ($\epsilon$)')
    ax2.set_title(r'MMSE as a function of $\alpha$')
    cb = plt.colorbar(p, ax=ax1) 
    #cb.set_ticks(numpy.array([0.3,0.4,0.5]))
    #cb.set_ticklabels(numpy.array([0.3,0.4,0.5]))

    ax1.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\phi$')

    l1, = ppl.plot(alphas, eps[:,1], label=r'$\phi = '+str(phis[1])+r'$',ax=ax2)
    c1 = l1.get_color()
    ppl.plot(alphas,stoc_eps_02, '.-', color=c1)
    
    indmin = numpy.argmin(eps[:,1])
    ppl.plot(alphas[indmin],eps[indmin,1],'o',color=c1)

    l2, = ppl.plot(alphas, eps[:,9], label=r'$\phi = '+str(phis[9])+r'$',ax=ax2)
    c2 = l2.get_color()
    ppl.plot(alphas,stoc_eps_10, '.-', color=c2)
    
    indmin = numpy.argmin(eps[:,9])
    ppl.plot(alphas[indmin],eps[indmin,9],'o',color=c2)
    
    l3, = ppl.plot(alphas, eps[:,-1], label=r'$\phi = '+str(phis[-1])+r'$',ax=ax2)
    c3 = l3.get_color()
    ppl.plot(alphas,stoc_eps_20, '.-', color=c3)
    
    indmin = numpy.argmin(eps[:,-1])
    ppl.plot(alphas[indmin],eps[indmin,-1],'o',color=c3)
   
    ppl.legend(ax2)

    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\epsilon$')

    plt.savefig("figure_5_7.eps")
