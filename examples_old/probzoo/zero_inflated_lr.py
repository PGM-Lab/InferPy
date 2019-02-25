import inferpy as inf
from inferpy.models import Normal, Beta, Categorical, Deterministic



d, N =  10, 200


#define the weights
w0 = Normal(0,1)
with inf.replicate(size=d):
    w = Normal(0, 1)


p = Beta(1,1)

# define the generative model
with inf.replicate(size=N):
    x = Normal(0, 1, observed=True, dim=d)
    y0 = Normal(inf.matmul(x,w), 1.0, observed=True)

    h = Categorical(probs=[p, 1-p])
    y1 = Deterministic(1.)

    # not working until issue #58 is solved
    y = Deterministic(inf.case({h.equal(0) : y0, h.equal(1) : y1}), observed = True)


h = Categorical(probs=[0.2,0.8])
h.probs
h.sample()





# toy data generation
x_train = Normal(loc=0, scale=1, dim=d).sample(N)
y_train = Normal(loc=5, scale=1, dim=1).sample(N)
data = {x.name: x_train, y.name: y_train}


# compile and fit the model with training data
m.compile()
m.fit(data)

m.posterior([w, w0])


x.dist.__matmul__(w.dist)

import tensorflow as tf

tf.matmul(x.dist,w.dist, transpose_a=False, transpose_b=True)