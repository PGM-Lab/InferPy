import tensorflow as tf
from tensorflow_probability.python import edward2 as ed


def sanitize_input_arg(arg):
    # This function sanitize the input arguments to create Random Variables. It does:
    # - convert Random Variables (from inferpy or edward2) to tensors, even if there
    # are in a list or nested list, in order to allow to use them as arguments for other Random Variables
    # - if hasattr dtype and is float64, cast to float32
    if isinstance(arg, list):
        return [sanitize_input_arg(nested_arg) for nested_arg in arg]

    # if not a list, sanitize the arg
    sanitized_arg = arg

    # if it is a random variable, convert it to tensor so it can be used as input arg by new random variable
    if isinstance(arg, (ed.RandomVariable, RandomVariable)):
        sanitized_arg = tf.convert_to_tensor(sanitized_arg)

    # if it has dtype arg, is float, and different from default float type (floatx), cast it
    if hasattr(sanitized_arg, "dtype") and util.common.is_float(sanitized_arg.dtype) and sanitized_arg.dtype != util.floatx():
        sanitized_arg = tf.cast(sanitized_arg, util.floatx())

    return sanitized_arg


from inferpy.contextmanager import datamodel  # noqa: F401
from .random_variable import RandomVariable
from .random_variable import *  # noqa: F403
from .prob_model import probmodel  # noqa: F401
from .parameter import Parameter  # noqa: F401
from inferpy import inference  # noqa: F401
from inferpy import util


__all__ = [
            'datamodel',
            'inference',
            'Parameter',
            'probmodel',
        ] + distributions_all + CUSTOM_RANDOM_VARIABLES  # noqa: F405
