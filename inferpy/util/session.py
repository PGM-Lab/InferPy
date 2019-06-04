
import tensorflow as tf
import warnings


"""
Module to manage global sessions and graphs executed in sessions
"""

__session = None


def new_session(gpu_memory_fraction=0.0):
    # Create a new session. By default do not use GPU. Use gpu_memory_fraction > 0 (and <= 1) to use GPU.
    if gpu_memory_fraction <= 0.0:
        set_session(tf.Session())
    else:
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction  # memory_fraction [0-1]
        set_session(tf.Session(config=config))


def get_session():
    global __session
    if not __session:
        __session = tf.Session()
    return __session


def set_session(session):
    global __session
    if __session:
        warnings.warn("Running session closed to use the provided session instead")
        __session.close()
    __session = session


def swap_session(new_session):
    global __session
    old_session = __session
    __session = new_session
    return old_session


def clear_session():
    global __session
    if __session:
        __session.close()
    __session = tf.Session()
