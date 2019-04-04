exec(open("./util.py").read())
assert "preprocess_data" in dir()


# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist

from tensorflow_probability import edward2 as ed
import matplotlib.pyplot as plt



from mpl_toolkits.mplot3d import Axes3D

tf.reset_default_graph()
sess = tf.Session()


####### load data  #########

(x_train, y_train), (x_test, y_test) = mnist.load_data()


img0 = x_train[0]


N = 500
D = [3, 10*10, 28 * 28]



C = list(range(0, 3))

x_train, y_train = preprocess_data(x_train, y_train, N, C, D)
x_test, y_test = preprocess_data(x_test, y_test, 500, C, D)

plot_imgs(x_train, 10,10)

k, d0, d1 = D


# definition of a generic model
@inf.probmodel
def pca(k,d0,d1):


    beta0 = inf.Normal(loc=tf.ones([k, d0]), scale=1.0, name="beta0")
    alpha0 = inf.Normal(loc=tf.ones([d0]), scale=1.0, name="alpha0")
    beta1 = inf.Normal(loc=tf.ones([d0, d1]), scale=1.0, name="beta1")
    alpha1 = inf.Normal(loc=tf.ones([d1]), scale=1.0, name="alpha1")


    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5, 1.0, name="z")
        h0 = tf.nn.relu(z @ beta0 + alpha0, name="h0")
        x = inf.Normal(h0 @ beta1 + alpha1 , 1.0, name="x")




@inf.probmodel
def qmodel(k,d0, d1):

    ### layer 0

    qbeta0_loc = inf.Parameter(tf.ones([k,d0])*0.5, name="qbeta0_loc")
    qbeta0_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d0])*0.5, name="qbeta0_scale"))

    qbeta0 = inf.Normal(qbeta0_loc, qbeta0_scale, name="beta0")

    ###

    qalpha0_loc = inf.Parameter(tf.ones([d0])*0.5, name="qalpha0_loc")
    qalpha0_scale = tf.math.softplus(inf.Parameter(tf.ones([d0])*0.5,name="qalpha0_scale"))

    alpha0 = inf.Normal(qalpha0_loc, qalpha0_scale, name="alpha0")

    ### layer 1

    qbeta1_loc = inf.Parameter(tf.ones([d0,d1])*0.5, name="qbeta1_loc")
    qbeta1_scale = tf.math.softplus(inf.Parameter(tf.ones([d0,d1])*0.5, name="qbeta1_scale"))

    qbeta1 = inf.Normal(qbeta1_loc, qbeta1_scale, name="beta1")

    ###

    qalpha1_loc = inf.Parameter(tf.ones([d1])*0.5, name="qalpha1_loc")
    qalpha1_scale = tf.math.softplus(inf.Parameter(tf.ones([d1])*0.5,name="qalpha1_scale"))

    alpha1 = inf.Normal(qalpha1_loc, qalpha1_scale, name="alpha1")



    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k])*0.5, name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k])*0.5, name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")



# define custom elbo function
def custom_elbo(pmodel, qmodel, sample_dict):
    # create combined model
    plate_size = pmodel._get_plate_size(sample_dict)

    qvars, _ = qmodel.expand_model(plate_size)

    with ed.interception(inf.util.random_variable.set_values(**{**qvars, **sample_dict})):
        pvars, _ = pmodel.expand_model(plate_size)

    # compute energy
    energy = tf.reduce_sum(pvars["x"].log_prob(pvars["x"].value)) + \
             tf.reduce_sum(pvars["z"].log_prob(pvars["z"].value)) + \
             tf.reduce_sum(pvars["alpha0"].log_prob(pvars["alpha0"].value)) + \
             tf.reduce_sum(pvars["beta0"].log_prob(pvars["beta0"].value)) + \
             tf.reduce_sum(pvars["alpha1"].log_prob(pvars["alpha1"].value)) + \
             tf.reduce_sum(pvars["beta1"].log_prob(pvars["beta1"].value))


    # compute entropy
    entropy = - (tf.reduce_sum(qvars["z"].log_prob(qvars["z"].value)) + \
            tf.reduce_sum(qvars["alpha0"].log_prob(qvars["alpha0"].value)) + \
             tf.reduce_sum(qvars["beta0"].log_prob(qvars["beta0"].value)) + \
             tf.reduce_sum(qvars["alpha1"].log_prob(qvars["alpha1"].value)) + \
             tf.reduce_sum(qvars["beta1"].log_prob(qvars["beta1"].value)))

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO


##### EXAMPLE OF USE



# create an instance of the model
m = pca(k,d0,d1)


optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
VI = inf.inference.VI(qmodel(k,d0,d1), epochs=1000, optimizer= optimizer, loss=custom_elbo)
m.fit({"x": x_train}, VI)


post = {k:sess.run(v.loc) for k,v in m.posterior.items()}




fig = plt.figure(figsize=plt.figaspect(0.5))
rotate = [0,120,240]

for r in range(0,3):

    ax = fig.add_subplot(1, len(rotate), r+1, projection='3d')

    for c in C:
        ax.view_init(30, rotate[r]+50)

        col = plt.cm.brg(c / (len(C) - 1))
        ax.scatter(post["z"][y_train == c, 0], post["z"][y_train == c, 1], post["z"][y_train == c, 2], color=col,
                   label=str(c), alpha=0.4)
plt.legend()

plt.show()

ax = fig.add_subplot(1,2,2, projection='3d')


#
#
# fig, ax = plt.subplots()
# for c in C:
#     col = plt.cm.brg(c / (len(C) - 1))
#     ax.scatter(post["z"][y_train == c, 0], post["z"][y_train == c, 1], color=col,
#                label=str(c), alpha=0.4)
#     plt.legend()
#
# plt.show()
#
#
# ax.legend()
# ax.grid(True)
#
#
# for c in C:
#     col = plt.cm.brg(c / (len(C) - 1))
#
#     ax.scatter(post["z"][y_train == c, 0], post["z"][y_train == c, 1], color=col,
#                label=str(c), alpha=0.4)
#     plt.legend()
#
# plt.show()




#
#
# # sample

z = post["z"]
beta0 = post["beta0"]
alpha0 = post["alpha0"]
beta1 = post["beta1"]
alpha1 = post["alpha1"]

h0 = tf.nn.relu(z @ beta0  + alpha0)
x_gen = inf.Normal(h0 @ beta1  + alpha1, 1, name="x").sample()


plot_imgs(x_gen, 5,5)


# 500 imgs of 28x28 pixels and a hidden layer of 100 nodes, gaussian ---> 81084x2 = 162168

m.vars["z"]

# np.sum([np.prod(v.shape.as_list()) for k,v in m.posterior.items()])