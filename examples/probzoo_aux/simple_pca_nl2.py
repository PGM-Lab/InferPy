# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed


k, d0, d1 = 1, 100, 2



@inf.probmodel
def pca(k, d0, d1, decoder):

    with inf.datamodel():
        z = inf.Normal(tf.ones([k])*0.5, 1., name="z")    # shape = [N,k]
        output = decoder(z,d0,d1)
        x = inf.Normal(output[:,:d1], tf.nn.softmax(output[:,d1:]), name="x")         # shape = [N,d]


def decoder(z,d0,d1):
    h0 = tf.layers.dense(z, d0, tf.nn.relu)
    return tf.layers.dense(h0, 2 * d1)



@inf.probmodel
def qmodel(k):
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
m = pca(k,d0,d1, decoder)

x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])

import matplotlib.pyplot as plt

plt.scatter(x_train[:,0], x_train[:,1])
plt.show()



VI = inf.inference.VI(qmodel(k), epochs=5000)
m.fit({"x": x_train}, VI)


post = m.posterior

sess.run(post["beta0"].loc)
sess.run(post["z"].loc)


z = post["z"]


##generate data
sess.run(decoder(z.loc, d0, d1))

