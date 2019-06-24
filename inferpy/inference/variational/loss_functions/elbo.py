import tensorflow as tf


def ELBO(pvars, qvars, batch_weight=1, **kwargs):
    """ Compute the loss tensor from the expanded variables of p and q models.
        Args:
            pvars (`dict<inferpy.RandomVariable>`): The dict with the expanded p random variables
            qvars (`dict<inferpy.RandomVariable>`): The dict with the expanded q random variables
            batch_weight (`float`): Weight to assign less importance to the energy, used when processing data in batches

        Returns (`tf.Tensor`):
            The generated loss tensor
    """

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
