# required packages
import numpy as np
import inferpy as inf
import tensorflow as tf

# number of observations
N = 1000

# Generate toy data
x_train = np.concatenate([
    inf.Normal([0.0, 0.0], scale=1.).sample(int(N/2)),
    inf.Normal([10.0, 10.0], scale=1.).sample(int(N/2))
    ])
x_test = np.concatenate([
    inf.Normal([0.0, 0.0], scale=1.).sample(int(N/2)),
    inf.Normal([10.0, 10.0], scale=1.).sample(int(N/2))
    ])


# definition of a generic model
@inf.probmodel
def pca(k, d):
    beta = inf.Normal(loc=tf.zeros([k, d]),
                      scale=1, name="beta")               # shape = [k,d]

    with inf.datamodel():
        z = inf.Normal(tf.ones(k), 1, name="z")       # shape = [N,k]
        x = inf.Normal(z @ beta, 1, name="x")         # shape = [N,d]


@inf.probmodel
def qmodel(k, d):
    qbeta_loc = inf.Parameter(tf.zeros([k, d]), name="qbeta_loc")
    qbeta_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d]),
                                                 name="qbeta_scale"))

    qbeta = inf.Normal(qbeta_loc, qbeta_scale, name="beta")

    with inf.datamodel():
        qz_loc = inf.Parameter(np.ones(k), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones(k),
                                                  name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")


# create an instance of the model and qmodel
m = pca(k=1, d=2)
q = qmodel(k=1, d=2)

# set the inference algorithm
VI = inf.inference.VI(q, epochs=2000)

# learn the parameters
m.fit({"x": x_train}, VI)

# extract the hidden encoding
hidden_encoding = m.posterior("z").parameters()["loc"]

# project x_test into the reduced space (encode)
m.posterior("z", data={"x": x_test}).sample(5)

# sample from the posterior predictive (i.e., simulate values for x given the learnt hidden)
m.posterior_predictive("x").sample(5)

# decode values from the hidden representation
m.posterior_predictive("x", data={"z": [2]}).sample(5)

