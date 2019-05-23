from tensorflow_probability import edward2 as ed
import tensorflow as tf

from inferpy import util


def ELBO(pmodel, qmodel, plate_size, batch_weight=1):
    # expand de qmodel
    qvars, _ = qmodel.expand_model(plate_size)

    # expand de pmodel, using the intercept.set_values function, to include the sample_dict and the expanded qvars
    with ed.interception(util.interceptor.set_values(**qvars)):
        pvars, _ = pmodel.expand_model(plate_size)

    # compute energy
    energy = tf.reduce_sum(
        [(batch_weight if p.is_datamodel else 1) * tf.reduce_sum(p.log_prob(p.value))
         for p in pvars.values()])

    # compute entropy
    entropy = - tf.reduce_sum(
        [(batch_weight if q.is_datamodel else 1) * tf.reduce_sum(q.log_prob(q.value))
         for q in qvars.values() if not q.is_datamodel])

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO
