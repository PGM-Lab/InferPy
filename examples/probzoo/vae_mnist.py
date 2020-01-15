import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import inferpy as inf

# number of components
k = 2
# size of the hidden layer in the NN
d0 = 100
# dimensionality of the data
dx = 28 * 28
# number of observations (dataset size)
N = 1000
# batch size
M = 100
# digits considered
DIG = [0, 1, 2]
# minimum scale
scale_epsilon = 0.01
# inference parameters
num_epochs = 1000
learning_rate = 0.01

# reset tensorflow
tf.reset_default_graph()
tf.set_random_seed(1234)



from inferpy.data import mnist

# load the data
(x_train, y_train), _ = mnist.load_data(num_instances=N, digits=DIG)

mnist.plot_digits(x_train, grid=[5,5])



############## Inferpy ##############

# P model and the  decoder NN
@inf.probmodel
def vae(k, d0, dx):
    with inf.datamodel():
        z = inf.Normal(tf.ones(k), 1,name="z")

        decoder = inf.layers.Sequential([
            tf.keras.layers.Dense(d0, activation=tf.nn.relu),
            tf.keras.layers.Dense(dx)])

        x = inf.Normal(decoder(z), 1, name="x")

# Q model for making inference
@inf.probmodel
def qmodel(k, d0, dx):
    with inf.datamodel():
        x = inf.Normal(tf.ones(dx), 1, name="x")

        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(d0, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * k)])

        output = encoder(x)
        qz_loc = output[:, :k]
        qz_scale = tf.nn.softplus(output[:, k:]) + scale_epsilon
        qz = inf.Normal(qz_loc, qz_scale, name="z")


#69



# Inference
############################

############## Inferpy ##############

m = vae(k, d0, dx)
q = qmodel(k, d0, dx)

# set the inference algorithm
SVI = inf.inference.SVI(q, epochs=1000, batch_size=M)


############################
############## Inferpy ##############

# learn the parameters
m.fit({"x": x_train}, SVI)

# Usage of the model with the inferred parameters
####################################################

############## Inferpy ##############
# extract the posterior and generate new digits
postz = np.concatenate([
    m.posterior("z", data={"x": x_train[i:i+M,:]}).sample()
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

##################
############## Inferpy ##############

x_gen = m.posterior_predictive('x', data={"z": postz[:M,:]}).sample()
mnist.plot_digits(x_gen, grid=[5,5])
#
#
# ##################
#
### Generate new images for a specific digit dig
dig = 0
postz_0 = postz[y_train == DIG[dig]]
# Less than plaze size, so we need to tile this up to 1000 instances
postz_0 = np.tile(postz_0, [int(np.ceil(N / postz_0.shape[0])), 1])[:N]
x_gen = m.posterior_predictive('x', data={"z": postz_0[:M]}).sample()
mnist.plot_digits(x_gen)


# Show how numbers are codified in the domain of the hidden variable z
# First define the range of the z domain
xaxis_min, yaxis_min = np.min(postz, axis=0)
xaxis_max, yaxis_max = np.max(postz, axis=0)
#
# generate 10x10 samples uniformly distributed, and get the first 1000
x = np.linspace(xaxis_min, xaxis_max, int(np.ceil(np.sqrt(N))))
y = np.linspace(yaxis_min, yaxis_max, int(np.ceil(np.sqrt(N))))
xx, yy = np.meshgrid(x, y)
postz = np.concatenate([np.expand_dims(xx.flatten(), 1), np.expand_dims(yy.flatten(), 1)], axis=1)[:N]

# Generate images for each point in the z variable domain
postx = np.concatenate([
    m.posterior_predictive('x', {"z": postz[i:i+M]}).sample()
    for i in range(0, N, M)])
# Get just 100 images, by steps of 10, so we can show 10x10 images
mnist.plot_digits(postx[::10], grid=[10, 10])
