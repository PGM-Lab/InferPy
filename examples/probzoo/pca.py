# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf

# definition of a generic model
@inf.probmodel
def pca(k,d):
    beta = inf.Normal(loc=np.zeros([k,d]),
                   scale=1, name="beta")               # shape = [k,d]

    with inf.datamodel():
        z = inf.Normal(np.ones([k]),1, name="z")    # shape = [N,k]
        x = inf.Normal(z @ beta , 1, name="x")         # shape = [N,d]


# create an instance of the model
m = pca(k=1,d=2)



@inf.probmodel
def qmodel(k,d):
    qbeta_loc = inf.Parameter(np.zeros([k,d]), name="qbeta_loc")
    qbeta_scale = tf.math.softplus(inf.Parameter(np.ones([k,d]),
                                                 name="qbeta_scale"))

    qbeta = inf.Normal(qbeta_loc, qbeta_scale, name="beta")

    with inf.datamodel():
        qz_loc = inf.Parameter(np.ones([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(np.ones([k]),
                                                  name="qz_scale"))

        qz = inf.Normal(qz_loc, qz_scale, name="z")



## example of use ###
# generate training data
N = 1000
sess = tf.Session()


x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])

import matplotlib.pyplot as plt

plt.scatter(x_train[:,0], x_train[:,1])
plt.show()

VI = inf.inference.VI(qmodel(k=1,d=2), epochs=5000)
m.fit({"x": x_train}, VI)

post = m.posterior

sess.run(post["beta"].loc)
sess.run(post["z"].loc)


z = post["z"]
beta = post["beta"]
x = inf.Normal(z @ beta , 1, name="x")


x_gen = x.sample()

plt.scatter(x_gen[:,0], x_gen[:,1])
plt.show()