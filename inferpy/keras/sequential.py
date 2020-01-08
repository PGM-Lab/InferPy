import tensorflow as tf
from inferpy import contextmanager


def Sequential(*args, **kwargs):
    model = tf.keras.Sequential(*args, **kwargs)
    # if the model is created inside a prob model, we need to  pass the sum(self.model.losses) to the
    # prob model, so it can be used by inference methods by includig the losses tensor in the optimizer

    # store this object in the layer_registry
    contextmanager.layer_registry.add_sequential(model)

    # and return the keras object
    return model
