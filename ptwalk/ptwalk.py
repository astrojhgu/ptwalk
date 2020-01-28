#!/usr/bin/env python
import numpy as np
from numpy import log, exp
from numpy.random import multinomial, uniform, normal, shuffle


class TWalkParam:
    def __init__(self, aw=1.5, at=6.0, pphi=0.1, fw=[0.4918, 0.4918, 0.0082 + 0.0082, 0.0]):
        # def __init__(self, aw=1.5, at=6.0, pphi=0.1, fw=[0.4918, 0.4918, 0., 0.]):
        self.aw = aw
        self.at = at
        self.pphi = pphi
        self.fw = fw/np.sum(fw)


def get_kernel(fw):
    while True:
        k1 = multinomial(1, fw) == 1
        k = np.where(k1 == 1)[0][0]
        if fw[k] != 0:
            break
    return k


def all_different(x1, x2):
    return np.all(x1 != x2)


def gen_phi(n, pphi):
    return uniform(size=n) < pphi


def sqr_norm(x):
    return np.sum(x**2)


def sim_walk(x, xp, param):
    aw = param.aw
    n = np.shape(x)[0]
    dx = x-xp
    phi = gen_phi(n, param.pphi)

    result = x.copy()
    for i in range(n):
        if phi[i]:
            u = uniform()
            z = aw/(1+aw)*(aw * u**2 + 2 * u - 1)
            result[i] = x[i]+dx[i]*z
    return (result, phi)


def sim_beta(param):
    at = param.at
    x = uniform()
    if x == 0:
        return x
    elif uniform() < (at-1)/(at*2):
        return exp(1.0/(at+1.0)*log(uniform()))
    else:
        return exp(1.0/(1.0-at)*log(uniform()))


def sim_traverse(x, xp, beta, param):
    n = np.shape(x)[0]
    result = x.copy()
    phi = gen_phi(n, param.pphi)
    for i in range(n):
        if phi[i]:
            result[i] = xp[i]+beta*(xp[i]-x[i])
    return (result, phi)


def sim_blow(x, xp, param):
    n = np.shape(x)[0]
    phi = gen_phi(n, param.pphi)
    dx = xp-x
    sigma = np.max(phi*np.abs(dx))
    result = x.copy()
    for i in range(n):
        if phi[i]:
            result[i] = xp[i]+sigma*normal()
    return (result, phi)


def g_blow_u(h, x, xp, phi):
    n = np.shape(x)[0]
    nphi = np.count_nonzero(phi)
    dx = xp-x
    sigma = np.max(phi*np.abs(dx))
    log2pi = np.log(2*np.pi)
    if nphi > 0:
        return (nphi/2.0)*log2pi + nphi*log(sigma) + 0.5*sqr_norm(h - xp)/(sigma**2)
    else:
        return 0


def sim_hop(x, xp, param):
    n = np.shape(x)[0]
    phi = gen_phi(n, param.pphi)
    dx = xp-x
    sigma = np.max(phi*np.abs(dx))/3.0
    result = x.copy()
    for i in range(n):
        if phi[i]:
            result[i] = x[i]+sigma*normal()
    return (result, phi)


def g_hop_u(h, x, xp, phi):
    return g_blow_u(h, x, xp, phi)


def calc_a(x, xp, up, yp, up_prop, phi, kernel, beta=None):
    nphi = np.count_nonzero(phi)
    if nphi == 0 or not all_different(yp, x):
        return 0.0
    elif kernel == 0:
        return exp(up_prop-up)
    elif kernel == 1:
        return exp((up_prop - up) + (nphi-2) * log(beta))
    elif kernel == 2:
        w1 = g_blow_u(yp, xp, x, phi)
        w2 = g_blow_u(xp, yp, x, phi)
        return exp((up_prop - up) + (w1 - w2))
    elif kernel == 3:
        w1 = g_hop_u(yp, xp, x, phi)
        w2 = g_hop_u(xp, yp, x, phi)
        return exp((up_prop - up) + (w1 - w2))
    else:
        assert(False)


def propose_move(x, xp, param):
    kernel = get_kernel(param.fw)
    beta = None
    if kernel == 0:
        (proposed, phi) = sim_walk(x, xp, param)
    elif kernel == 1:
        beta = sim_beta(param)
        (proposed, phi) = sim_traverse(x, xp, beta, param)
    elif kernel == 2:
        (proposed, phi) = sim_blow(x, xp, param)
    elif kernel == 3:
        (proposed, phi) = sim_hop(x, xp, param)
    else:
        assert(False)
    return (proposed, phi, kernel, beta)


def sample(flogprob, ensemble, old_logprob_list, param):
    nwalkers = len(ensemble)
    assert(nwalkers % 2 == 0)
    walker_id = [i for i in range(nwalkers)]
    shuffle(walker_id)
    group1 = walker_id[:nwalkers//2]
    group2 = walker_id[nwalkers//2:]
    proposed_points = [propose_move(ensemble[i1], ensemble[i2], param) for (
        i1, i2) in zip(group1, group2)]
    log_prob_list = [flogprob(x[0]) for x in proposed_points]

    for (i1, i2, up_prop, (yp, phi, k, beta)) in zip(group1, group2, log_prob_list, proposed_points):
        a = calc_a(ensemble[i2], ensemble[i1],
                   old_logprob_list[i1], yp, up_prop, phi, k, beta)
        if uniform() < a:
            ensemble[i1] = yp
            old_logprob_list[i1] = up_prop
