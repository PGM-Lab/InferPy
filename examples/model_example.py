import inferpy as inf
import pandas as pd
import edward as ed
import numpy as np
import tensorflow.contrib.distributions as tfd

N, K, d = 100, 5, 10



# variables
beta = inf.models.MultivariateNormalDiag(loc=np.zeros(K), scale_diag=np.ones(K))

with inf.replicate(size = N):
    z =  inf.models.Normal(loc=0, scale=1)
    x = inf.models.Bernoulli(beta, 1, dim=d, observed=True)


inf.models.Multinomial()


# model definition
m = inf.ProbModel(varlist=[beta,z,x])
m.compile()


# infer the parameters from data
data = pd.read_csv("inferpy/datasets/test.csv")
m.fit(data)


print(m.posterior(beta).loc)


"""
>>> x.shape
[1000, 10]
>>> x.shape
[1000, 10]
>>> x.dim
10
>>> x.batches
1000
>>> beta.loc
array([0.], dtype=float32)
>>> beta.scale
array([1.], dtype=float32)

"""


