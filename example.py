#!/usr/bin/env python

import numpy as np
from numpy.random import multinomial, uniform, normal, shuffle
import ptwalk
import matplotlib.pylab as plt
def rosenbrock(x):
    result=0
    for i in range(np.shape(x)[0]-1):
        result+=100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    return -result

def normal_dist(x):
    return -sum(x**2/2.0)

ndims=2
lp=rosenbrock
ensemble=[normal(size=ndims) for i in range(10)]
ensemble=[np.array(x) for x in ensemble]
logprob_list=[lp(x) for x in ensemble]

param=ptwalk.TWalkParam()
param.pphi=0.5

thin=100

xlist=[]
ylist=[]

for i in range(100000):
    ptwalk.sample(lp, ensemble, logprob_list, param)
    if i%thin==0:
        print(i)
        if i>1000:
            xlist.append(ensemble[0][0])
            ylist.append(ensemble[0][1])

print(np.std(xlist), np.std(ylist))
plt.plot(xlist, ylist, '.')
plt.show()
