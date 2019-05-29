# required pacakges
import inferpy as inf
import numpy as np
import tensorflow as tf


@inf.probmodel
def log_reg(d):
    w0 = inf.Normal(0., 1, name="w0")
    w = inf.Normal(0., 1, batch_shape=[d,1], name="w")

    with inf.datamodel():
        x = inf.Normal(0., 2., batch_shape=d, name="x")
        y = inf.Bernoulli(logits = w0 + x @ w, name="y")




@inf.probmodel
def qmodel(d):
    qw0_loc = inf.Parameter(0., name="qw0_loc")
    qw0_scale = tf.math.softplus(inf.Parameter(1., name="qw0_scale"))
    qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")

    qw_loc = inf.Parameter(tf.zeros([d,1]), name="qw_loc")
    qw_scale = tf.math.softplus(inf.Parameter(tf.ones([d,1]), name="qw_scale"))
    qw = inf.Normal(qw_loc, qw_scale, name="w")


# create an instance of the model
m = log_reg(d=2)

### create toy data
N = 1000
data = m.sample(size = N, data={"w0":0, "w":[[2],[1]]})
x_train = data["x"]
y_train = data["y"]



VI = inf.inference.VI(qmodel(2), epochs=10000)
m.fit({"x": x_train, "y":y_train}, VI)

sess = inf.get_session()
print(m.posterior["w"].sample())
print(sess.run(m.posterior["w"].loc))
