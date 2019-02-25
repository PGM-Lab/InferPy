import edward as ed
import inferpy as inf
import numpy as np
import tensorflow as tf



K, d, N, T = 3, 4, 1000, 5000


# toy data generation
x_train = np.vstack([inf.models.Normal(loc=0, scale=1, dim=d).sample(300),
                     inf.models.Normal(loc=10, scale=1, dim=d).sample(700)])

######## Inferpy ##########


# model definition
with inf.ProbModel() as m:

    # prior distributions
    with inf.replicate(size=K):
        mu = inf.models.Normal(loc=0, scale=1, dim=d)
        sigma = inf.models.InverseGamma(concentration=1, rate=1, dim=d,)
    p = inf.models.Dirichlet(np.ones(K)/K)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Categorical(probs = p)
        x = inf.models.Normal(mu[z], sigma[z],observed=True, dim=d)

# compile and fit the model with training data
data = {x: x_train}
m.compile(infMethod="MCMC")
m.fit(data)

# print the posterior
print(m.posterior(mu))

