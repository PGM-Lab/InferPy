import inferpy as inf
import numpy as np
from inferpy.util.wrappers import def_ProbModel




# toy data generation
x_train = np.vstack([inf.models.Normal(loc=0, scale=1, dim=10).sample(300),
                     inf.models.Normal(loc=10, scale=1, dim=10).sample(700)])


####### define your custom parameterizable model #####


@def_ProbModel
def custom_gaussian_mixture(K,d,N):
    # prior distributions
    with inf.replicate(size=K):
        mu = inf.models.Normal(loc=0, scale=1,dim=d)
        sigma = inf.models.InverseGamma(concentration=1, rate=1, dim=d,)
    p = inf.models.Dirichlet(np.ones(K)/K)

    # define the generative model
    with inf.replicate(size=N):
        z = inf.models.Categorical(probs = p)
        x = inf.models.Normal(mu[z], sigma[z],observed=True, dim=d)


##### create instances of this model ####


m1 = custom_gaussian_mixture(K=3,d=10,N=1000)
m2 = custom_gaussian_mixture(K=2,d=10,N=1000)

#### compile each model ####

m1.compile()
m2.compile(infMethod="MCMC")

#### fit the data with each model #####

m1.fit({m1.observed_vars[0] : x_train})
m2.fit({m2.observed_vars[0] : x_train})




#############################################
#############################################

## you could also use the predefined models ##


from inferpy.models.predefined import *

m3 = pca(K=3,d=10,N=1000)

m3.compile()

m3.fit({m3.observed_vars[0] : x_train})






