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
m_sample = inf.ProbModel(varlist=[w,z,x])
x_train = m_sample.sample(1000)

####

m = inf.ProbModel(varlist=[w,z,x])
m.compile(infMethod="KLqp")
m.fit(x_train)
m.posterior(z)

##


qw = inf.Qmodel.Normal(w)
qz = inf.Qmodel.Normal(z)

qmodel = inf.Qmodel([qw, qz])

m.compile(infMethod="KLqp", Q=qmodel)
m.fit(x_train)
m.posterior(z)

