import edward as ed
import inferpy as inf
import numpy as np
import tensorflow as tf



K, d, N = 3, 10, 1000

# model definition
with inf.ProbModel() as m:

    # Prior
    with inf.replicate(size=K):
        mu = inf.models.Normal(loc=0, scale=1, dim=d, name="mu")
        sigma = inf.models.InverseGamma(concentration=1, rate=1, dim=d, name="sigma")

    p = inf.models.Dirichlet(np.ones(K))



    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Categorical(probs = p, name="z")
        x = inf.models.Normal(mu[z], sigma[z], observed=True, dim=d, name="x")
        #x = inf.models.Normal(tf.reshape(inf.gather(mu,z),(N,d)),
        #                      tf.reshape(inf.gather(sigma,z),(N,d)), observed=True, dim=d, name="x")

        #x = inf.models.Normal(0, 1, observed=True, dim=d)

# toy data generation

k_train = inf.models.Categorical(logits=[0,0,0])
x_train = np.concatenate((inf.models.Normal(loc=0, scale=1, dim=d).sample(100),
                          inf.models.Normal(loc=10, scale=1, dim=d).sample(300),
                          inf.models.Normal(loc=20, scale=1, dim=d).sample(600)))
data = {x.name: x_train}


# compile and fit the model with training data
m.compile()
m.fit(data)


print(m.posterior(mu).loc)
print(m.posterior(p))
