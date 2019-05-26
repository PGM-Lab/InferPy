# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from inferpy.datasets import mnist

tf.reset_default_graph()
tf.set_random_seed(1234)

# number of components
k = 2
# dimensionality of the data
d = 28*28
# number of observations (dataset size)
N = 1000
# digits considered
DIG = [0,1,2]

learning_rate = 0.01


# definition of a generic model
@inf.probmodel
def pca(k,d):
    w = inf.Normal(loc=tf.zeros([k,d]),
                   scale=1, name="w")               # shape = [k,d]

    w0 = inf.Normal(loc=tf.zeros([d]),
                    scale=1, name="w0")  # shape = [d]

    with inf.datamodel():

        z = inf.Normal(tf.zeros([k]),1, name="z")       # shape = [N,k]
        x = inf.Normal( z @ w + w0, 1, name="x")         # shape = [N,d]



@inf.probmodel
def qmodel(k,d):
    qw_loc = inf.Parameter(tf.zeros([k,d]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([k,d]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")

    qw0_loc = inf.Parameter(tf.ones([d]), name="qw0_loc")
    qw0_scale = tf.math.softplus(inf.Parameter(tf.ones([d]), name="qw0_scale"))
    qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")


    with inf.datamodel():


        qz_loc = inf.Parameter(np.zeros([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        qz = inf.Normal(qz_loc, qz_scale, name="z")



# create an instance of the P model and the Q model
m = pca(k,d)
q = qmodel(k,d)

# load the data
(x_train, y_train), _ = mnist.load_data(num_instances=N, digits=DIG)

optimizer = tf.train.AdamOptimizer(learning_rate)
VI = inf.inference.VI(q, optimizer=optimizer, epochs=2000)

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
post = {v:sess.run(m.posterior[v].loc) for v in ["z", "w", "w0"] }


m.posterior["z"].copy()

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



