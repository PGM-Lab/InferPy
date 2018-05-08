import inferpy as inf
from inferpy.models import Normal

K, d, N = 5, 10, 200


# K defines the number of components.
K=10

# d defines the number of dimensions
d=20


#Prior for the principal components
with inf.replicate(size = K):
    w = Normal(loc = 0, scale = 1, dim = d)



###

# Number of observations
N = 1000

# define the generative model
with inf.replicate(size=N):
    z = Normal(0, 1, dim=K)
    x = Normal(inf.matmul(z,w), 1.0, observed=True, dim=d)


###

from inferpy import ProbModel

# Define the model
pca = ProbModel(vars = [w,z,x])

# Compile the model
pca.compile(infMethod = 'KLqp')



###


from inferpy import ProbModel

# Define the model
pca = ProbModel(vars = [w,z,x])

# Compile the model
pca.compile(infMethod = 'Laplace')


###

data = pca.sample(size=100)