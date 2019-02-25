import inferpy as inf

# K defines the number of components.
K=10

# d defines the number of dimensions
d=20

#Prior for the principal components
with inf.replicate(size = K):
    w = inf.models.Normal(loc = 0, scale = 1, dim = d)

# Number of observations
N = 1000

# define the generative model
with inf.replicate(size=N):
    z = inf.models.Normal(0, 1, dim=K)
    x = inf.models.Normal(inf.matmul(z,w), 1.0, observed=True, dim=d)


##############

m = inf.ProbModel(varlist=[w,z,x])
m.compile()

##

data = m.sample(1000)
log_like = m.log_prob(data)
sum_log_like = m.sum_log_prob(data)