# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


d = 2
N = 2000


@inf.probmodel
def logregression(d, N, w_init=(1, 1), x_init=(0, 1)):

    w = inf.Normal(loc=np.ones(d, dtype="float32") * w_init[0], scale=1. * w_init[1], name="w")
    w0 = inf.Normal(loc=1. * w_init[0], scale=1. * w_init[1], name="w0")

    with inf.datamodel():
        x = inf.Normal(loc=np.ones(d, dtype="float32") * x_init[0], scale=1. * x_init[1], name="x")
        y = inf.Bernoulli(logits=tf.tensordot(x, w, axes=[[1], [0]]) + w0, name="y")


# create an instance of the model for sampling
w_sampling = [2, 1]
w0_sampling = 0

m = logregression(d, N, x_init=(2, 10))

training = m.prior(
    ['x', 'y'],
    data={"w0": w0_sampling, "w": w_sampling},
    size_datamodel=N).sample()

x_train = training['x']
y_train = training['y']

# show the data distribution according to its class
plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='blue', alpha=0.1)
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='red', alpha=0.1)

plt.axis([-20, 30, -20, 30])
plt.show()


# train the model using MCMC and the previous data
num_samples = 1000

mcmc = inf.inference.MCMC()
m.fit(training, mcmc)


# generate posterior data to check the learnt distribution
testing = m.posterior_predictive(['x', 'y']).sample(num_samples)

x_test = testing['x']
y_test = testing['y']

# and show the data distribution according to its class
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color='blue', alpha=0.1)
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color='red', alpha=0.1)

plt.axis([-20, 30, -20, 30])
plt.show()
