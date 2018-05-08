import numpy as np
import inferpy as inf
from inferpy.models import Normal, InverseGamma, Dirichlet

# K defines the number of components.
K=10

# d defines the number of dimensions
d=20

#Prior for the principal components
with inf.replicate(size = K):
    mu = Normal(loc = 0, scale = 1, dim = d)

