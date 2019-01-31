import inferpy as inf


inf.models.ALLOWED_VARS

# Bernoulli: binary distribution with probability p and 1-p

x = inf.models.Bernoulli(probs=0.5)

# or

x = inf.models.Bernoulli(logits=0)

if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
#################   15
# Beta: continuous distribution defined in the interval [0,1]
# parametrized by two positive shape parameters, denoted alpha and beta

x = inf.models.Beta(concentration0=0.5, concentration1=0.5)

# or simply:

x = inf.models.Beta(0.5,0.5)

if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
# Categorical: discrete distribution

x = inf.models.Categorical(probs=[0.5,0.5])

# or

x = inf.models.Categorical(logits=[0,0])

if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
# Deterministic

x = inf.models.Deterministic(loc=5)

# or simply:

x = inf.models.Deterministic(5)
if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
############# 43

# Dirichlet: continuous multivariate probability distributions parameterized by a vector
#  of positive reals. It is a multivariate generalization of the beta distribution. Dirichlet
# distributions are commonly used as prior distributions in Bayesian statistics

x = inf.models.Dirichlet(concentration=[5,1])

# or simply:

x = inf.models.Dirichlet([5,1])
if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
#############55
# Exponential:
x = inf.models.Exponential(rate=1)

# or simply

x = inf.models.Exponential(1)
if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
# Gamma:



x = inf.models.Gamma(concentration=3, rate=2)

# Inverse Gamma: NOT WORKING



x = inf.models.InverseGamma(concentration=3, rate=2)

###########75

# Laplace: the Laplace distribution is a continuous probability distribution


x = inf.models.Laplace(loc=0, scale=1)

# or simply

x = inf.models.Laplace(0,1)


if x.dist is None: raise Exception(x.__class__.__name__+" is empty")

# Multinomial

# The multinomial is a discrete distribution

x = inf.models.Multinomial(total_count=4, probs=[0.5,0.5])

# or

x = inf.models.Multinomial(total_count=4, logits=[0,0])


if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
# Normal



x = inf.models.Normal(loc=0, scale=1)

# or

x = inf.models.Normal(0,1)

if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
########112
# Poisson


x = inf.models.Poisson(rate=4)

# or

x = inf.models.Poisson(4)

if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
####### 123

# Uniform


x = inf.models.Uniform(low=1, high=3)

# or

inf.models.Uniform(1,3)


if x.dist is None: raise Exception(x.__class__.__name__+" is empty")
#136



x = inf.models.MultivariateNormalDiag(
    loc=[1., -1],
    scale_diag=[1, 2.]
)



if x.dist is None: raise Exception(x.__class__.__name__+" is empty")