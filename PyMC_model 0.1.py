# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:20:21 2023

@author: yavry
"""

import pymc as pm
import numpy as np

NSubjects = 10  # replace with actual number of subjects
TrialStart = 1  
TrialEnd = 900  
v = np.random.rand(TrialEnd, NSubjects)  # replace with actual v values
u = np.random.rand(TrialEnd, NSubjects)  # replace with actual u values
y = np.random.rand(TrialEnd, NSubjects)  # replace with actual y values

with pm.Model() as model:
    A1mu = pm.Normal('A1mu', mu=0.0, sigma=1.0E3)
    A1prec = pm.Gamma('A1prec', alpha=1.0E-3, beta=1.0E-3)
    B1mu = pm.Normal('B1mu', mu=0.0, sigma=1.0E3)
    B1prec = pm.Gamma('B1prec', alpha=1.0E-3, beta=1.0E-3)

    A1 = pm.Normal('A1', mu=A1mu, sigma=(1/A1prec), shape=NSubjects)
    B1 = pm.Normal('B1', mu=B1mu, sigma=(1/B1prec), shape=NSubjects)
    A = pm.Deterministic('A', pm.math.invlogit(A1))
    B = pm.Deterministic('B', pm.math.invlogit(B1))

    q = pm.Gamma('q', alpha=1.0E-3, beta=1.0E-3, shape=NSubjects)
    r = pm.Gamma('r', alpha=1.0E-3, beta=1.0E-3, shape=NSubjects)

    x = pm.Normal('x', mu=0.0, sigma=1.0E3, shape=(TrialEnd+1, NSubjects))

    for s in range(NSubjects):
        for t in range(TrialStart, TrialEnd):
            xmu = A[s] * x[t, s] - v[t, s] * B[s] * u[t, s]
            pm.Potential(f'xmu_{t}_{s}', pm.Normal.dist(mu=xmu, sigma=(1/q[s])))
            pm.Potential(f'y_{t}_{s}', pm.Normal.dist(mu=x[t, s], sigma=(1/r[s])))

    trace = pm.sample(2000, tune=1000)
