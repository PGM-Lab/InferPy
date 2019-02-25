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


# define custom elbo function
def custom_elbo(pmodel, qvars, sample_dict):
    # create combined model
    plate_size = pmodel._get_plate_size(sample_dict)
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


## example of use ###
# generate training data
N = 1000
sess = tf.Session()
x_train = sess.run(ed.Normal(5., 2.).distribution.sample(N))


m = simple()

VI = models.inference.VI(q_model, loss=custom_elbo, epochs=5000)

m.fit({"x": x_train}, VI)

