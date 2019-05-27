# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from inferpy.datasets import mnist



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
def nlpca(k, d0, dx, decoder):
    with inf.datamodel():
        z = inf.Normal(tf.ones([k]) * 0.1, 1., name="z")  # shape = [N,k]
        output = decoder(z, d0, dx)
        x_loc = output[:, :dx]
        x_scale = tf.nn.softmax(output[:, dx:]) + scale_epsilon
        x = inf.Normal(x_loc, x_scale, name="x")  # shape = [N,d]


def decoder(z, d0, dx):  # k -> d0 -> 2*dx
    h0 = tf.layers.dense(z, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * dx)


@inf.probmodel
def qmodel(k):
    with inf.datamodel():
        qz_loc = inf.Parameter(np.ones([k])*0.1, name="qz_loc")
        qz_scale = tf.math.softmax(inf.Parameter(tf.ones([k]), name="qz_scale"))
        qz = inf.Normal(qz_loc, qz_scale, name="z")




m = nlpca(k, d0, dx, decoder)
q = qmodel(k)

# set the inference algorithm
VI = inf.inference.VI(q, epochs=5000)

(x_train, y_train), _ = mnist.load_data(num_instances=N, digits=DIG)

# plot the digits
mnist.plot_digits(x_train)

# learn the parameters
m.fit({"x": x_train}, VI)


# Plot the evolution of the loss

L = VI.losses
plt.plot(range(len(L)), L)

plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss evolution')
plt.grid(True)
plt.show()


#extract the hidden encoding
sess = inf.get_session()
post = {"z":sess.run(m.posterior["z"].loc)}


# plot
markers = ["x", "+", "o"]
colors = [plt.get_cmap("gist_rainbow")(0.05),
          plt.get_cmap("gnuplot2")(0.08),
          plt.get_cmap("gist_rainbow")(0.33)]
transp = [0.9, 0.9, 0.5]

fig = plt.figure()

for c in range(0, len(DIG)):
    col = colors[c]
    plt.scatter(post["z"][y_train == DIG[c], 0], post["z"][y_train == DIG[c], 1], color=col,
                label=DIG[c], marker=markers[c], alpha=transp[c], s=60)
    plt.legend()

plt.show()



with inf.contextmanager.observe(m.posterior, {"z": post["z"]}):
    x_gen = m._last_expanded_vars['x'].sample()

mnist.plot_digits(x_gen, grid=[10,5])

