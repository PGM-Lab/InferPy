# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

# definition of a generic model
@inf.probmodel
def pca(k,d0):

    beta0 = inf.Normal(loc=tf.ones([k, d0])*0.5,
                   scale=1.0, name="beta0")               # shape = [k,d]

    alpha0 = inf.Normal(loc=tf.ones([d0])*0.5,
                      scale=1.0, name="alpha0")  # shape = [k,d]


    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5,1.0, name="z")    # shape = [N,k]
        x = inf.Normal(z @ beta0 + alpha0 , 1.0, name="x")         # shape = [N,d]




@inf.probmodel
def qmodel(k,d0):
    qbeta0_loc = inf.Parameter(tf.ones([k,d0])*0.5, name="qbeta0_loc")
    qbeta0_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d0])*0.5,
                                                 name="qbeta0_scale"))

    qbeta0 = inf.Normal(qbeta0_loc, qbeta0_scale, name="beta0")


    qalpha0_loc = inf.Parameter(tf.ones([d0])*0.5, name="qalpha0_loc")
    qalpha0_scale = tf.math.softplus(inf.Parameter(tf.ones([d0])*0.5,
                                                 name="qalpha0_scale"))

    alpha0 = inf.Normal(qalpha0_loc, qalpha0_scale, name="alpha0")


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
m = pca(k=1,d0=2)

x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])

import matplotlib.pyplot as plt

plt.scatter(x_train[:,0], x_train[:,1])
plt.show()




#### inference with fit

VI = inf.inference.VI(qmodel(k=1,d0=2), epochs=5000)
m.fit({"x": x_train}, VI)

post = m.posterior

sess.run(post["beta0"].loc)
sess.run(post["z"].loc)


z = post["z"]
beta0 = post["beta0"]
alpha0 = post["alpha0"]
x = inf.Normal(z @ beta0  + alpha0, 1, name="x")


x_gen = x.sample()

plt.scatter(x_gen[:,0], x_gen[:,1])
plt.show()


##

from inferpy.inference.loss_functions.elbo import *

m = pca(k=1,d0=2)
q = qmodel(k=1,d0=2)
loss_tensor = ELBO(m,q, {"x": x_train})


with tf.Session() as sess:

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss_tensor)

    sess.run(tf.global_variables_initializer())

    t = []
    for i in range(0,100):
        sess.run(train)
        t += [sess.run(loss_tensor)]
        print(t[-1])


