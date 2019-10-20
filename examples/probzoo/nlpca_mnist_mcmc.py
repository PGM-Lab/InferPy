# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from inferpy.data import mnist


# number of components
k = 2
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 28 * 28
# number of observations (dataset size)
N = 1000

# digits considered
DIG = [0, 1, 2]

scale_epsilon = 0.01


@inf.probmodel
def nlpca(k, d0, dx, decoder):
    with inf.datamodel():
        z = inf.Normal(tf.ones(k) * 0.1, 1., name="z")  # shape = [N,k]
        output = decoder(z, d0, dx)
        x_loc = output[:, :dx]
        x_scale = tf.nn.softmax(output[:, dx:]) + scale_epsilon
        x = inf.Normal(x_loc, x_scale, name="x")  # shape = [N,d]


# initial values
loc_init = 0.001
scale_init = 1

def decoder(z, d0, dx):  # k -> d0 -> 2*dx

    beta0 = inf.Normal(tf.ones([k, d0]) * loc_init, scale_init, name="beta0")
    alpha0 = inf.Normal(tf.ones([d0]) * loc_init, scale_init, name="alpha0")

    h0 = tf.nn.relu(z @ beta0 + alpha0, name="h0")

    ######

    beta1 = inf.Normal(tf.ones([d0, 2*dx]) * loc_init, scale_init, name="beta1")
    alpha1 = inf.Normal(tf.ones([2*dx]) * loc_init, scale_init, name="alpha1")

    output = z @ beta0 + alpha0

    return output


(x_train, y_train), _ = mnist.load_data(num_instances=N, digits=DIG)

m = nlpca(k, d0, dx, decoder)
inf_method = inf.inference.MCMC()

# learn the parameters
m.fit({"x": x_train}, inf_method)


# Plot the evolution of the loss

L = inf_method.losses
plt.plot(range(len(L)), L)

plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Loss evolution')
plt.grid(True)
plt.show()


# posterior sample from the hidden variable z, given the training data
sess = inf.get_session()
postz = m.posterior("z", data={"x": x_train}).sample()


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


# Generate new images
x_gen = m.posterior_predictive('x').sample()
mnist.plot_digits(x_gen)


# Generate new images for a specific number
NUM = 0
postz_0 = postz[y_train == DIG[NUM]]
# Less than plaze size, so we need to tile this up to 1000 instances
postz_0 = np.tile(postz_0, [int(np.ceil(N / postz_0.shape[0])), 1])[:N]
x_gen = m.posterior_predictive('x', data={"z": postz_0}).sample()
mnist.plot_digits(x_gen)


# Show how numbers are codified in the domain of the hidden variable z
# First define the range of the z domain
xaxis_min, yaxis_min = np.min(postz, axis=0)
xaxis_max, yaxis_max = np.max(postz, axis=0)

# generate 10x10 samples uniformly distributed, and get the first 1000
x = np.linspace(xaxis_min, xaxis_max, int(np.ceil(np.sqrt(N))))
y = np.linspace(yaxis_min, yaxis_max, int(np.ceil(np.sqrt(N))))
xx, yy = np.meshgrid(x, y)
postz = np.concatenate([np.expand_dims(xx.flatten(), 1), np.expand_dims(yy.flatten(), 1)], axis=1)[:N]

# Generate images for each point in the z variable domain
postx = m.posterior_predictive('x', {"z": postz}).sample()
# Get just 100 images, by steps of 10, so we can show 10x10 images
mnist.plot_digits(postx[::10], grid=[10, 10])
