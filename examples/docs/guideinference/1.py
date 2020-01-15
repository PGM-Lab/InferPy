import inferpy as inf
import numpy as np
import tensorflow as tf


@inf.probmodel
def pca(k,d):
    w = inf.Normal(loc=tf.zeros([k,d]), scale=1, name="w")      # shape = [k,d]
    with inf.datamodel():
        z = inf.Normal(tf.ones([k]),1, name="z")                # shape = [N,k]
        x = inf.Normal(z @ w , 1, name="x")                     # shape = [N,d]






#### data (do not show)
# generate training data
N = 1000
sess = tf.Session()
x_train = np.concatenate([inf.Normal([0.0,0.0], scale=1.).sample(int(N/2)), inf.Normal([10.0,10.0], scale=1.).sample(int(N/2))])



### encapsulated inference

#28


@inf.probmodel
def qmodel(k,d):
    qw_loc = inf.Parameter(tf.ones([k,d]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([k, d]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")

    with inf.datamodel():
        qz_loc = inf.Parameter(tf.ones([k]), name="qz_loc")
        qz_scale = tf.math.softplus(inf.Parameter(tf.ones([k]), name="qz_scale"))
        qz = inf.Normal(qz_loc, qz_scale, name="z")



# set the inference algorithm
VI = inf.inference.VI(qmodel(k=1,d=2), epochs=1000)
# create an instance of the model
m = pca(k=1,d=2)
# run the inference
m.fit({"x": x_train}, VI)

"""
 0 epochs	 44601.14453125....................
 200 epochs	 44196.98046875....................
 400 epochs	 50616.359375....................
 600 epochs	 41085.6484375....................
 800 epochs	 30349.79296875....................
 
"""

print(m.posterior("w").parameters())
"""
>>> m.posterior("w").parameters()
{'name': 'w',
 'allow_nan_stats': True,
 'validate_args': False,
 'scale': array([[0.9834974 , 0.99731755]], dtype=float32),
 'loc': array([[1.7543027, 1.7246702]], dtype=float32)}
"""

# 70


## ELBO definition


# define custom elbo function
def custom_elbo(pvars, qvars, **kwargs):

    # compute energy
    energy = tf.reduce_sum([tf.reduce_sum(p.log_prob(p.value)) for p in pvars.values()])

    # compute entropy
    entropy = - tf.reduce_sum([tf.reduce_sum(q.log_prob(q.value)) for q in qvars.values()])

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO



# set the inference algorithm
VI = inf.inference.VI(qmodel(k=1,d=2), loss=custom_elbo, epochs=1000)

# run the inference
m.fit({"x": x_train}, VI)

# set the inference algorithm
MC = inf.inference.MCMC()
# run the inference
m.fit({"x": x_train}, MC)

#104

# extract the posterior of z
hidden_encoding = m.posterior("z").parameters()["samples"].mean(axis=0)



SVI = inf.inference.SVI(qmodel(k=1,d=2), epochs=1000, batch_size=200)