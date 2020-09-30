# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from inferpy.data import mnist


# number of observations (dataset size)
N = 1000

# batch size
M = 100


@inf.probmodel
def digit_classifier(k, d0, dx, dy):
    with inf.datamodel():
        z = inf.Normal(tf.ones(k) * 0.1, 1., name="z")

        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(d0, tf.nn.relu),
            tf.keras.layers.Dense(dx)
        ])
        classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(dy)
        ])
        x = inf.Normal(decoder(z), 1., name="x")
        y = inf.Categorical(logits=classifier(z), name="y")

p = digit_classifier(k=2, d0=100, dx=28*28, dy=3)


@inf.probmodel
def qmodel(k, d0, dx):
    with inf.datamodel():
        x = inf.Normal(tf.ones(dx), 1, name="x")

        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(d0, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * k)
        ])
        output = encoder(x)
        qz_loc = output[:, :k]
        qz_scale = tf.nn.softplus(output[:, k:])+0.01
        qz = inf.Normal(qz_loc, qz_scale, name="z")

q = qmodel(k=2, d0=100, dx=28*28)

# get the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data(
    num_instances=N, digits=[0, 1, 2])
# set the inference algorithm
SVI = inf.inference.SVI(q, epochs=10000, batch_size=M)
# fit the model to the data
p.fit({"x": x_train, "y":y_train}, SVI)


# Plot the evolution of the loss
L = SVI.losses
plt.plot(range(len(L)), L)

plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss evolution')
plt.grid(True)
plt.show()


# extract the posterior of z given the training data
postz = np.concatenate([
    p.posterior("z", data={"x": x_train[i:i + M, :]}).sample()
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



# predict a set of images
def predict(x):
    postz = p.posterior("z", data={"x": x}).sample()
    return p.posterior_predictive("y", data={"z":postz}).sample()

y_gen = predict(x_test[:M])

# compute the accuracy
acc = np.sum(y_test[:M] == y_gen)/M
print(f"accuracy: {acc}")