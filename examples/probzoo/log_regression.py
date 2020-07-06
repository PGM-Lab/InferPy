### Setting up

import inferpy as inf
import numpy as np
import tensorflow as tf

d = 2
N = 10000



### Model definition ####

@inf.probmodel
def log_reg(d):
    w0 = inf.Normal(0., 1., name="w0")
    w = inf.Normal(np.zeros([d, 1]), np.ones([d, 1]), name="w")

    with inf.datamodel():
        x = inf.Normal(np.zeros(d), 2., name="x")  # the scale is broadcasted to shape [d] because of loc
        y = inf.Bernoulli(logits=w0 + x @ w, name="y")


@inf.probmodel
def qmodel(d):
    qw0_loc = inf.Parameter(0., name="qw0_loc")
    qw0_scale = tf.math.softplus(inf.Parameter(1., name="qw0_scale"))
    qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")

    qw_loc = inf.Parameter(tf.zeros([d, 1]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([d, 1]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")




##### Sample from prior model

# instance of the model
m = log_reg(d)

# create toy data
data = m.prior(["x", "y"], data={"w0": 0, "w": [[2], [1]]}).sample(N)
x_train = data["x"]
y_train = data["y"]



#### Inference

VI = inf.inference.VI(qmodel(d), epochs=10000)
m.fit({"x": x_train, "y": y_train}, VI)


#### Usage of the inferred model


# Print the parameters
w_post = m.posterior("w").parameters()["loc"]
w0_post = m.posterior("w0").parameters()["loc"]

print(w_post, w0_post)

# Sample from the posterior
post_sample = m.posterior_predictive(["x","y"], data={"w":w_post, "w":w0_post}).sample()
x_gen = post_sample["x"]
y_gen = post_sample["y"]

print(x_gen, y_gen)





##### Plot the results

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
for c in [0, 1]:
    x_gen_c = x_gen[y_gen.flatten() == c, :]
    plt.plot(x_gen_c[:, 0], x_gen_c[:, 1], 'bx' if c == 0 else 'rx')
plt.show()


#### 86