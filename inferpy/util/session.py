
import tensorflow as tf
import warnings


"""
Module to manage global sessions and graphs executed in sessions
"""

__session = None


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
