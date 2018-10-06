import inferpy as inf
import numpy as np
from inferpy.util.wrappers import def_ProbModel



@def_ProbModel
def gaussian_mixture(K,d,N):
    # prior distributions
    with inf.replicate(size=K):
        mu = inf.models.Normal(loc=0, scale=1,dim=d)
        sigma = inf.models.InverseGamma(concentration=1, rate=1, dim=d,)
    p = inf.models.Dirichlet(np.ones(K)/K)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Categorical(probs = p)
        x = inf.models.Normal(mu[z], sigma[z],observed=True, dim=d)





@def_ProbModel
def linear_regression(d,N):

    #define the weights
    w0 = inf.models.Normal(0,1)
    w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = inf.models.Normal(0, 1, observed=True, dim=d)
        y = inf.models.Normal(w0 + inf.dot(x,w), 1.0, observed=True)





@def_ProbModel
def log_regression(d,N):

    #define the weights
    w0 = inf.models.Normal(0,1)
    w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = inf.models.Normal(0, 1, observed=True, dim=d)
        y = inf.models.Bernoulli(logits=w0+inf.dot(x, w), observed=True)





@def_ProbModel
def log_regression(K,d,N):

    #define the weights
    w0 = inf.models.Normal(0,1, dim=K)

    with inf.replicate(size=K):
        w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = inf.models.Normal(0, 1, observed=True, dim=d)
        y = inf.models.Bernoulli(logits = w0 + inf.matmul(x, w, transpose_b=True), observed=True)



@def_ProbModel
def pca(K,d,N):
    #define the weights
    with inf.replicate(size=K):
        w = inf.models.Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Normal(0, 1, dim=K)
        x = inf.models.Normal(inf.matmul(z,w),
                               1.0, observed=True, dim=d)



@def_ProbModel
def pca_with_ard_prior(K,d,N):
    #define the weights
    with inf.replicate(size=K):
        w = inf.models.Normal(0, 1, dim=d)

    sigma = inf.models.InverseGamma(1.0,1.0)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Normal(0, 1, dim=K)
        x = inf.models.Normal(inf.matmul(z,w),
                   sigma, observed=True, dim=d)