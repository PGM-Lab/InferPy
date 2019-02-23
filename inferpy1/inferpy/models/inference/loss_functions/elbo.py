from tensorflow_probability import edward2 as ed
import tensorflow as tf

from inferpy import util


def ELBO(pmodel, qvars, sample_dict):
    # NOTE: Energy can be computed using inferpy RV, but need to use p.log_prob(p.value, tf_run=False).
    # NOTE: The interception requires to use ed2.vars, otherwise cause nans when learning.
    # NOTE: Entropy cannot be done in this way. There is something with interception which requires to be the same rv
    # For now, until we get more understanding about the underneath process, use ed2 RVs to compute the loss function

    # transform intro ed2 RVs
    qvars = {k: v.var for k, v in qvars.items()}

    # create combined model
    plate_size = pmodel._get_plate_size(sample_dict)
    with ed.interception(util.random_variable.set_values(**{**qvars, **sample_dict})):
        pvars, _ = pmodel.expand_model(plate_size)

    # transform intro ed2 RVs
    pvars = {k: v.var for k, v in pvars.items()}

    # compute energy
    energy = tf.reduce_sum([tf.reduce_sum(p.distribution.log_prob(p.value)) for p in pvars.values()])

    # compute entropy
    entropy = - tf.reduce_sum([tf.reduce_sum(q.distribution.log_prob(q.value)) for q in qvars.values()])

    # compute ELBO
    ELBO = energy + entropy

    # This function will be minimized. Return minus ELBO
    return -ELBO
