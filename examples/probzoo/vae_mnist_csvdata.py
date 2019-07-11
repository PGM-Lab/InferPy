# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from inferpy.data import mnist
from inferpy.data.loaders import CsvLoader



# number of components
k = 2
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 28*28
# number of observations (dataset size)
N = 1000

# digits considered
DIG = [0,1,2]

scale_epsilon = 0.01


@inf.probmodel
def vae(k, d0, dx, decoder):
    with inf.datamodel():
        z = inf.Normal(tf.ones([k]) * 0.5, 1., name="z")  # shape = [N,k]
        output = decoder(z, d0, dx)
        x_loc = output[:, :dx]
        x_scale = tf.nn.softmax(output[:, dx:]) + scale_epsilon
        x = inf.Normal(x_loc, x_scale, name="x")  # shape = [N,d]


def decoder(z, d0, dx):  # k -> d0 -> 2*dx
    h0 = tf.layers.dense(z, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * dx)


# Q-model  approximating P
def encoder(x, d0, k):  # dx -> d0 -> 2*k
    h0 = tf.layers.dense(x, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * k)


@inf.probmodel
def qmodel(k, d0, dx, encoder):
    with inf.datamodel():
        x = inf.Normal(tf.ones([dx]), 1, name="x")

        output = encoder(x, d0, k)
        qz_loc = output[:, :k]
        qz_scale = tf.nn.softmax(output[:, k:])
        qz = inf.Normal(qz_loc, qz_scale, name="z")


m = vae(k, d0, dx, decoder)
q = qmodel(k, d0, dx, encoder)

# set the inference algorithm
SVI = inf.inference.SVI(q, epochs=200, batch_size=100)


# create data loader from 10 csv files
path = [f"./mnist_xtrain{i}.csv" for i in range(10)]

data_loader = CsvLoader(path, variables={"x" : range(dx)})
m.fit(data_loader.to_dict(), SVI)


# Plot the evolution of the loss

L = SVI.losses
plt.plot(range(len(L)), L)

plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss evolution')
plt.grid(True)
plt.show()

