from tensorflow_probability import edward2 as ed
import tensorflow as tf

from inferpy import util


def ELBO(pmodel, qmodel, sample_dict, plate_size=None):
    # create combined model; for that first compute the plate size (for now, just one plate can be used)
    if not plate_size:
        plate_size = pmodel._get_plate_size(sample_dict)

    # expand the qmodel (just in case the q model uses data from sample_dict, use interceptor too)
    with ed.interception(util.interceptor.set_values(**sample_dict)):
        qvars, _ = qmodel.expand_model(plate_size)

    # expand de pmodel, using the intercept.set_values function, to include the sample_dict and the expanded qvars
    with ed.interception(util.interceptor.set_values(**{**qvars, **sample_dict})):
        pvars, _ = pmodel.expand_model(plate_size)

    # compute energy
    energy = tf.reduce_sum([tf.reduce_sum(p.log_prob(p.value)) for p in pvars.values()])

    # compute entropy
    entropy = - tf.reduce_sum([tf.reduce_sum(q.log_prob(q.value)) for q in qvars.values()])

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO
