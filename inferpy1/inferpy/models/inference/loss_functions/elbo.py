from tensorflow_probability import edward2 as ed
import tensorflow as tf

from inferpy import util


def ELBO(pmodel, qvars, sample_dict):
    # create combined model
    plate_size = pmodel._get_plate_size(sample_dict)
    with ed.interception(util.random_variable.set_values(**{**qvars, **sample_dict})):
        pvars, _ = pmodel.expand_model(plate_size)

    # compute energy
    energy = tf.reduce_sum([tf.reduce_sum(p.log_prob(p.value)) for p in pvars.values()])

    # compute entropy
    entropy = - tf.reduce_sum([tf.reduce_sum(q.log_prob(q.value)) for q in qvars.values()])

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO
