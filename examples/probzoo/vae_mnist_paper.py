import tensorflow as tf
import inferpy as inf
from inferpy.data import mnist
import tensorflow_probability as tfp

N, M = 1000, 100 # data and batch size
(x_train, _), _ = mnist.load_data(num_instances=N,
                                  digits=[0, 1, 2])


# P model and the  decoder NN
@inf.probmodel
def vae(k, d0, d):
    with inf.datamodel():
        z = inf.Normal(tf.ones(k), 1,name="z")
        decoder = inf.layers.Sequential([
            tfp.layers.DenseFlipout(d0, activation=tf.nn.relu),
            tf.keras.layers.Dense(d)])
        x = inf.Normal(decoder(z), 1, name="x")
p = vae(k=2, d0=100, d=28*28)


# Q model and the encoder NN
@inf.probmodel
def qmodel(k, d0, d):
    with inf.datamodel():
        x = inf.Normal(tf.ones(d), 1, name="x")
        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(d0, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * k)])
        output = encoder(x)
        qz_loc = output[:, :k]
        qz_scale = tf.nn.softplus(output[:, k:])+0.01
        qz = inf.Normal(qz_loc, qz_scale, name="z")
q = qmodel(k=2, d0=100, d=28*28)

# set the inference algorithm
SVI = inf.inference.SVI(q, epochs=1000, batch_size=100)

# learn the parameters
p.fit({"x": x_train}, SVI)

# extract the posterior and generate new digits
postz = p.posterior("z", data={"x": x_train[:M,:]}).sample()
x_gen = p.posterior_predictive('x', data={"z": postz}).sample()





######## not shown in the paper

import numpy as np
import matplotlib.pyplot as plt

DIG=[0, 1, 2]

(x_train, y_train), _ = mnist.load_data(num_instances=N,
                                  digits=[0, 1, 2])

# extract the posterior and generate new digits
postz = np.concatenate([
    p.posterior("z", data={"x": x_train[i:i+M,:]}).sample()
    for i in range(0,N,M)])

# for each input instance, plot the hidden encoding coloured by the number that it represents
markers = ["x", "+", "o"]
colors = [plt.get_cmap("gist_rainbow")(0.05),
          plt.get_cmap("gnuplot2")(0.08),
          plt.get_cmap("gist_rainbow")(0.33)]
transp = [0.9, 0.9, 0.5]

fig = plt.figure()

for c in range(0, len(DIG)):
    col = colors[c]
    plt.scatter(postz[y_train == DIG[c], 0], postz[y_train == DIG[c], 1], color=col,
                label=DIG[c], marker=markers[c], alpha=transp[c], s=60)
    plt.legend()

plt.show()


mnist.plot_digits(x_gen, grid=[5,5])

