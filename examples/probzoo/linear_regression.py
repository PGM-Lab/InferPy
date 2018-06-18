import edward as ed
import inferpy as inf
from inferpy.models import Normal
import numpy as np

d, N =  5, 20000

# model definition
with inf.ProbModel() as m:

    #define the weights
    w0 = Normal(0,1)
    #with inf.replicate(size=d):
    w = Normal(0, 1, dim=d)

    # define the generative model
    with inf.replicate(size=N):
        x = Normal(0, 1, observed=True, dim=d)
        y = Normal(w0 + inf.matmul(x,w, transpose_b=True), 1.0, observed=True)

# toy data generation
x_train = inf.models.Normal(loc=10, scale=5, dim=d).sample(N)
y_train = x_train/10 + inf.models.Normal(loc=0, scale=5, dim=d).sample(N)




data = {x.name: x_train, y.name: y_train}
data

# compile and fit the model with training data
m.compile()
m.fit(data)

print(m.posterior([w, w0]))


