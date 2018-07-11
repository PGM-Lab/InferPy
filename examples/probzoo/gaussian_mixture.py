import edward as ed
import inferpy as inf
import numpy as np



K, d, N = 3, 10, 500

# model definition
with inf.ProbModel() as m:

    # Prior
    with inf.replicate(size=K):
        mu = inf.models.Normal(loc=0, scale=1, dim=d)
        sigma = inf.models.InverseGamma(concentration=[1], rate=[1], dim=d)

    p = inf.models.Dirichlet([1], dim=K)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Categorical(logits = p)
        x = inf.models.Normal(inf.gather(mu,z), scale=inf.gather(sigma,z), observed=True)


    mu[z].shape


# toy data generation
x_train = Normal(loc=0, scale=1, dim=d).sample(N)
y_train = Bernoulli(probs=[0.4]).sample(N)
data = {x.name: x_train, y.name: np.reshape(y_train, (N,1))}


# compile and fit the model with training data
m.compile()
m.fit(data)

print(m.posterior([w, w0]))


