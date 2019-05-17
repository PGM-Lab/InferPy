# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

# definition of a generic model
@inf.probmodel
def pca(k,d0,d1):

    beta0 = inf.Normal(loc=tf.ones([k, d0])*0.5,
                   scale=1.0, name="beta0")

    alpha0 = inf.Normal(loc=tf.ones([d0])*0.5,
                      scale=1.0, name="alpha0")

    beta1 = inf.Normal(loc=tf.ones([d0, d1])*0.5,
                   scale=1.0, name="beta1")


    alpha1 = inf.Normal(loc=tf.ones([d1])*0.5,
                      scale=1.0, name="alpha1")



    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5,1.0, name="z")    # shape = [N,k]
        h0 = tf.nn.relu(z @ beta0 + alpha0)
        x = inf.Normal(h0 @ beta1 + alpha1 , 1.0, name="x")         # shape = [N,d]




@inf.probmodel
def qmodel(k,d0, d1):
    qbeta0_loc = inf.Parameter(tf.ones([k,d0])*0.5, name="qbeta0_loc")
    qbeta0_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d0])*0.5,
                                                 name="qbeta0_scale"))
    qbeta0 = inf.Normal(qbeta0_loc, qbeta0_scale, name="beta0")


    qalpha0_loc = inf.Parameter(tf.ones([d0])*0.5, name="qalpha0_loc")
    qalpha0_scale = tf.math.softplus(inf.Parameter(tf.ones([d0])*0.5,
                                                 name="qalpha0_scale"))
    alpha0 = inf.Normal(qalpha0_loc, qalpha0_scale, name="alpha0")

    ###

    qbeta1_loc = inf.Parameter(tf.ones([d0,d1])*0.5, name="qbeta1_loc")
    qbeta1_scale = tf.math.softplus(inf.Parameter(tf.ones([d0,d1])*0.5,
                                                 name="qbeta1_scale"))
    qbeta1 = inf.Normal(qbeta1_loc, qbeta1_scale, name="beta1")


    qalpha1_loc = inf.Parameter(tf.ones([d1])*0.5, name="qalpha1_loc")
    qalpha1_scale = tf.math.softplus(inf.Parameter(tf.ones([d1])*0.5,
                                                 name="qalpha1_scale"))
    alpha1 = inf.Normal(qalpha1_loc, qalpha1_scale, name="alpha1")



    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k])*0.5, name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]),
                                                  name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")



## example of use ###
# generate training data
N = 1000
sess = tf.Session()



# create an instance of the model
m = pca(k=1,d0=2, d1=2)

x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])

import matplotlib.pyplot as plt

plt.scatter(x_train[:,0], x_train[:,1])
plt.show()

from tensorflow_probability import edward2 as ed




# define custom elbo function
def custom_elbo(pmodel, qmodel, sample_dict):
    # create combined model
    plate_size = pmodel._get_plate_size(sample_dict)

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





VI = inf.inference.VI(qmodel(k=1,d0=2, d1=2), epochs=5000, loss=custom_elbo)
m.fit({"x": x_train}, VI)


post = m.posterior

sess.run(post["beta0"].loc)
sess.run(post["z"].loc)


z = post["z"]
beta0 = post["beta0"]
alpha0 = post["alpha0"]
beta1 = post["beta1"]
alpha1 = post["alpha1"]

h0 = tf.nn.relu(z @ beta0  + alpha0)
x = inf.Normal(h0 @ beta1  + alpha1, 1, name="x")


x_gen = x.sample()

plt.scatter(x_gen[:,0], x_gen[:,1])
plt.show()