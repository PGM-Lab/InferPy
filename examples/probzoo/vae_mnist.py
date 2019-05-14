#!wget https://raw.githubusercontent.com/PGM-Lab/deep_prob_modeling/master/notebooks/src/util.py
exec(open("./util.py").read())
assert "preprocess_data" in dir()


# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from inferpy import util

from tensorflow_probability import edward2 as ed


from mpl_toolkits.mplot3d import Axes3D


tf.reset_default_graph()

####### load data  #########

(x_train, y_train), (x_test, y_test) = mnist.load_data()

N = 1000
k = 3
D = [10 * 10, 28 * 28]



C = list(range(0, 3))

x_train, y_train = preprocess_data(x_train, y_train, N, C, D)
x_test, y_test = preprocess_data(x_test, y_test, 500, C, D)

plot_imgs(x_train, 10,10)

#######


# definition of a generic model
@inf.probmodel
def vae(k,d0, d1):

    beta0 = inf.Normal(tf.ones([d0, k]) * 0.5, tf.ones([d0, k]) * 0.5, name="beta0")
    alpha0 = inf.Normal(tf.ones([d0]) * 0.5, tf.ones([d0]) * 0.5, name="alpha0")
    beta1 = inf.Normal(tf.ones([d1, d0]) * 0.5, tf.ones([d1, d0]) * 0.5, name="beta1")
    alpha1 = inf.Normal(tf.ones([d1]) * 0.5, tf.ones([d1]) * 0.5, name="alpha1")

    with inf.datamodel():
        z = inf.Normal(tf.ones([k]),1.0, name="z")
        h0 = tf.nn.relu(z @ tf.transpose(beta0) + alpha0)
        h1 = h0 @ tf.transpose(beta1) + alpha1
        x = inf.Normal(h1, 1.0, name="x")



m = vae(k=3,d0=D[0], d1=D[1])
m.expand_model(size=1000)



@inf.probmodel
def qmodel(k,d0, d1):

    eps = tf.constant(0.0000001, dtype="float32")

    qbeta0_loc = inf.Parameter(tf.ones([d0,k])*0.5, name="qbeta0_loc")
    qbeta0_scale = tf.math.softplus(inf.Parameter(tf.ones([d0, k])*0.5, name="qbeta0_scale")) + eps
    qbeta0 = inf.Normal(qbeta0_loc, qbeta0_scale, name="beta0")

    qalpha0_loc = inf.Parameter(tf.ones([d0])*0.5, name="qalpha0_loc")
    qalpha0_scale = tf.math.softplus(inf.Parameter(tf.ones([d0])*0.5, name="qalpha0_scale"))+ eps
    qalpha0 = inf.Normal(qalpha0_loc, qalpha0_scale, name="alpha0")

    qbeta1_loc = inf.Parameter(tf.ones([d1, d0])*0.5, name="qbeta1_loc")
    qbeta1_scale = tf.math.softplus(inf.Parameter(tf.ones([d1, d0])*0.5, name="qbeta1_scale"))+ eps
    qbeta1 = inf.Normal(qbeta1_loc, qbeta1_scale, name="beta1")

    qalpha1_loc = inf.Parameter(tf.ones([d1])*0.5, name="qalpha1_loc")
    qalpha1_scale = tf.math.softplus(inf.Parameter(tf.ones([d1])*0.5, name="qalpha1_scale"))+ eps
    qalpha1 = inf.Normal(qalpha1_loc, qalpha1_scale, name="alpha1")


    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k])*100,
                                                  name="qz_scale")) + eps

        qz = inf.Normal(qz_loc, qz_scale, name="z")




# define custom elbo function
def custom_elbo(pmodel, qmodel, sample_dict):
    # create combined model
    plate_size = util.iterables.get_plate_size(pmodel.vars, sample_dict)

    qvars, _ = qmodel.expand_model(plate_size)

    with ed.interception(inf.util.random_variable.set_values(**{**qvars, **sample_dict})):
        pvars, _ = pmodel.expand_model(plate_size)

    # compute energy
    energy = tf.reduce_sum([tf.reduce_sum(p.log_prob(p.value)) for p in pvars.values()])


    # compute entropy
    entropy = - tf.reduce_sum([tf.reduce_sum(q.log_prob(q.value)) for q in qvars.values()])

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO


# create an instance of the model
m = vae(k=3,d0=D[0], d1=D[1])




VI = inf.inference.VI(qmodel(k=3,d0=D[0], d1=D[1]), epochs=5000, loss=custom_elbo)
m.fit({"x": x_train}, VI)

x_train.shape

sess = tf.Session()

post = {k:sess.run(v.loc) for k,v in m.posterior.items()}


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c in C:
    ax.view_init(30, 45)

    col = plt.cm.brg(c / (len(C) - 1))
    ax.scatter(post["z"][y_train == c, 0], post["z"][y_train == c, 1], post["z"][y_train == c, 2], color=col,
               label=str(c), alpha=0.4)
    plt.legend()

plt.show()



#
#
# ### sample with
#
#
# beta0 = m.posterior["beta0"].value
# alpha0 = m.posterior["alpha0"].value
# beta1 = m.posterior["beta1"].value
# alpha1 = m.posterior["alpha1"].value
# z = m.posterior["z"].value
#
# h0 = tf.nn.relu(z @ tf.transpose(beta0) + alpha0)
# h1 = h0 @ tf.transpose(beta1) + alpha1
# x = inf.Normal(h1, 1., name="x")
#
# x_gen = x.sample()
#
#
# plot_imgs(x_gen, nx=5, ny=5)