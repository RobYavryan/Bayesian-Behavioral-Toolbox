import pymc as pm
import numpy as np

NSubjects = 10  # replace with actual number of subjects
TrialStart = 1  
TrialEnd = 900  

q = 2
r = 10
A = 0.9
B = 0.2
v = np.randint(1, TrialEnd-TrialStart)
p = [np.zeros[180], 45*np.ones[600], np.zeros[180]] # This wants to be a 1-D numpy array of the perturbations you want

x[0] = np.normrnd(0, q)
y[0] = np.normrnd(x[0], r)
for i in range(TrialStart, TrialEnd):
    x[i] = np.normrnd(A*x[i-1] + v[i-1]*B*u[i-1] ,q)
    y[i] = np.normrnd(x[i], r)
    u[i] = y[i] + p[i]

# v = np.random.rand(TrialEnd, NSubjects)  # replace with actual v values
# u = np.random.rand(TrialEnd, NSubjects)  # replace with actual u values
# y = np.random.rand(TrialEnd, NSubjects)  # replace with actual y values

with pm.Model() as model:
    A1mu = pm.Normal('A1mu', mu=2, sigma=1)
    A1prec = pm.Gamma('A1prec', mu=2, sigma=1)
    B1mu = pm.Normal('B1mu', mu=-1, sigma=0.75)
    B1prec = pm.Gamma('B1prec', mu=-0.75, sigma=0.75)

    A1 = pm.Normal('A1', mu=A1mu, sigma=(1/A1prec), shape=NSubjects)
    B1 = pm.Normal('B1', mu=B1mu, sigma=(1/B1prec), shape=NSubjects)
    A = pm.Deterministic('A', pm.math.invlogit(A1))
    B = pm.Deterministic('B', pm.math.invlogit(B1))

    q = pm.Gamma('q', alpha=1.0E-3, beta=1.0E-3, shape=NSubjects) # Same idea with mean around 1 or 2 and std around 1 or 2
    r = pm.Gamma('r', alpha=1.0E-3, beta=1.0E-3, shape=NSubjects) # Same idea mean around 10 and std aroung 5 or more


    for s in range(NSubjects):
        for t in range(TrialStart, TrialEnd):
            xmu = A[s] * x[t, s] - v[t, s] * B[s] * u[t, s]
            pm.Potential(f'xmu_{t}_{s}', pm.Normal.dist(mu=xmu, sigma=(1/q[s])))
            pm.Potential(f'y_{t}_{s}', pm.Normal.dist(mu=x[t, s], sigma=(1/r[s])))

    trace = pm.sample(2000, tune=1000)
