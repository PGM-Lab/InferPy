import tensorflow as tf
from tensorflow_probability import edward2 as ed
import inferpy as inf

from inferpy import models


@models.probmodel
def simple():
    theta = models.Normal(0, 1, name="theta")
    with models.datamodel():
        x = models.Normal(theta, 2, name="x")


@models.probmodel
def q_model():
    qtheta_loc = models.Parameter(1., name="qtheta_loc")
    qtheta_scale = tf.math.softplus(models.Parameter(1., name="qtheta_scale"))

    qtheta = models.Normal(qtheta_loc, qtheta_scale, name="theta")



## example of use ###
# generate training data
N = 1000
sess = tf.Session()
x_train = sess.run(ed.Normal(5., 2.).distribution.sample(N))

##

m = simple()

VI = models.inference.VI(q_model(), epochs=5000)

m.fit({"x": x_train}, VI)

