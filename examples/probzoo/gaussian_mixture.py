import edward as ed
import inferpy as inf
import numpy as np
import tensorflow as tf



K, d, N = 2, 1, 1000

# model definition
with inf.ProbModel() as m:

    # Prior
    with inf.replicate(size=K):
        mu = inf.models.Normal(loc=0, scale=1, dim=d, name="mu")
      #  sigma = inf.models.InverseGamma(concentration=1, rate=1, dim=d, name="sigma")
        sigma = inf.models.Normal(loc=0, scale=1, dim=d, name="sigma")


    p = inf.models.Dirichlet(np.ones(K)/K)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Categorical(probs = p, name="z")
        x = inf.models.Normal(mu[z], sigma[z], observed=True, dim=d, name="x", allow_nan_stats=False)
 #       x = inf.models.Normal(tf.gather(mu.dist, tf.reshape(z.dist, (N,)), validate_indices=True),
  #                            tf.gather(sigma.dist, tf.reshape(z.dist, (N,)), validate_indices=True),
   #                           observed=True, dim=d, name="x")



# toy data generation
x_train = np.vstack([inf.models.Normal(loc=0, scale=1, dim=d).sample(300),
                     inf.models.Normal(loc=10, scale=1, dim=d).sample(700)])

data = {x.name: x_train}


m.compile()
m.fit(data)


ed.MAP([mu.dist, sigma.dist, p.dist, z.dist], data={x.dist:x_train})



