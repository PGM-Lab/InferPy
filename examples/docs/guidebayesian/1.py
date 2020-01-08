import numpy as np
import inferpy as inf
N=1000
x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])
##########


# required pacakges
import inferpy as inf
import tensorflow as tf


# number of components
k = 1
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 2
# number of observations (dataset size)
N = 1000


@inf.probmodel
def nlpca(k, d0, dx, decoder):

    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5, 1., name="z")    # shape = [N,k]
        output = decoder(z,d0,dx)
        x_loc = output[:,:dx]
        x_scale = tf.nn.softmax(output[:,dx:])
        x = inf.Normal(x_loc, x_scale, name="x")   # shape = [N,d]


def decoder(z,d0,dx):
    h0 = tf.layers.dense(z, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * dx)


# Q-model  approximating P

@inf.probmodel
def qmodel(k):
    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k])*0.5, name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]),name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")


# create an instance of the model
m = nlpca(k,d0,dx, decoder)

# set the inference algorithm
VI = inf.inference.VI(qmodel(k), epochs=5000)

# learn the parameters
m.fit({"x": x_train}, VI)

#extract the hidden representation
hidden_encoding = m.posterior("z")
print(hidden_encoding.sample())

#63




def decoder_keras(z,d0,dx):
    h0 = tf.keras.layers.Dense(d0, activation=tf.nn.relu)
    h1 = tf.keras.layers.Dense(2*dx)
    return h1(h0(z))

# create an instance of the model
m = nlpca(k,d0,dx, decoder_keras)
m.fit({"x": x_train}, VI)




#80


def decoder_seq(z,d0,dx):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d0, activation=tf.nn.relu),
        tf.keras.layers.Dense(2 * dx)
    ])(z)

# create an instance of the model and fit the data
m = nlpca(k,d0,dx, decoder_seq)
m.fit({"x": x_train}, VI)



# 95

import tensorflow_probability as tfp

def decoder_bayesian(z,d0,dx):
    return inf.layers.Sequential([
        tfp.layers.DenseFlipout(d0, activation=tf.nn.relu),
        tfp.layers.DenseLocalReparameterization(d0, activation=tf.nn.relu),
        tfp.layers.DenseReparameterization(d0, activation=tf.nn.relu),
        tf.keras.layers.Dense(2 * dx)
    ])(z)


# create an instance of the model
m = nlpca(k,d0,dx, decoder_bayesian)
m.fit({"x": x_train}, VI)

