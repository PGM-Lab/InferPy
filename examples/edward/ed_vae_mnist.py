import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability.python.edward2 as ed
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


# Model definition
######################


############## Edward ##############

def vae(k, d0, dx, N):
    z = ed.Normal(loc=tf.ones(k), scale=1., sample_shape=N, name="z")

    decoder = inf.layers.Sequential([
        tf.keras.layers.Dense(d0, activation=tf.nn.relu, name="h0"),
        tf.keras.layers.Dense(dx, name="h1")],
    name="decoder")
    x = ed.Normal(loc=decoder(z, d0, dx), scale=1., name="x")
    return z, x



# Q model for making inference which is parametrized by the data x.
def qmodel(k, d0, x):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(d0, activation=tf.nn.relu, name="h0"),
        tf.keras.layers.Dense(2 * k, name="h1")],
    name = "encoder")
    output = encoder(x)
    qz_loc = output[:, :k]
    qz_scale = tf.nn.softplus(output[:, k:]) + scale_epsilon
    qz = ed.Normal(loc=qz_loc, scale=qz_scale, name="qz")
    return qz




#68

# Inference
############################

### preparing inference and batched data













############## Edward ##############

batch = tf.data.Dataset.from_tensor_slices(x_train)\
        .shuffle(M)\
        .batch(M)\
        .repeat()\
        .make_one_shot_iterator().get_next()

qz = qmodel(k, d0, batch)

with ed.interception(ed.make_value_setter(z=qz, x=batch)):
    pz, px = vae(k, d0, dx, M)

energy = N/M*tf.reduce_sum(pz.distribution.log_prob(pz.value)) + \
         N/M*tf.reduce_sum(px.distribution.log_prob(px.value))
entropy = N/M*tf.reduce_sum(qz.distribution.log_prob(qz.value))

elbo = energy - entropy


############################


############## Edward ##############

sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(-elbo)
init = tf.global_variables_initializer()
sess.run(init)

t = []
for i in range(num_epochs + 1):
    for j in range(N // M):
        elbo_ij, _ = sess.run([elbo, train])

        t.append(elbo_ij)
        if j == 0 and i % 200 == 0:
            print("\n {} epochs\t {}".format(i, t[-1]), end="", flush=True)
        if j == 0 and i % 20 == 0:
            print(".", end="", flush=True)


# Usage of the model with the inferred parameters
####################################################

### posterior sample from the hidden variable z, given the training data

############## Edward ##############

def get_tfvar(name):
    for v in tf.trainable_variables():
        if str.startswith(v.name, name):
            return v

def predictive_nn(x, beta0, alpha0, beta1, alpha1):
    h0 = tf.nn.relu(x @ beta0 + alpha0)
    output = h0 @ beta1 + alpha1

    return output

weights_encoder = [sess.run(get_tfvar("encoder/h" + name)) for name in ["0/kernel", "0/bias", "1/kernel", "1/bias",]]
postz = sess.run(predictive_nn(x_train, *weights_encoder)[:, :k])



# plot

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

### Generate new images

############## Edward ##############

weights_decoder = [sess.run(get_tfvar("decoder/h" + name)) for name in ["0/kernel", "0/bias", "1/kernel", "1/bias",]]
x_gen = sess.run(predictive_nn(postz, *weights_decoder)[:, :dx])

nx, ny = (3,3)
fig, ax = plt.subplots(nx, ny, figsize=(12, 12))
fig.tight_layout(pad=0.3, rect=[0, 0, 0.9, 0.9])
for x, y in [(i, j) for i in list(range(nx)) for j in list(range(ny))]:
    img_i = x_gen[x + y * nx].reshape((28, 28))
    i = (y, x) if nx > 1 else y
    ax[i].imshow(img_i, cmap='gray')
plt.show()





tf.trainable_variables()