""" Obtained from Keras GitHub repository: https://github.com/keras-team/keras/blob/master/keras/backend/common.py

"""

_FLOATX = "float32"


def floatx():
    """Returns the default float type, as a string.  (e.g. float16, float32, float64).

    Returns:
        String: the current default float type.

    Example:
        >>> inf.floatx()
        'float32'


    """
    return _FLOATX


def set_floatx(floatx):
    """ Sets the default float type.

    Args:
        floatx: String, 'float16', 'float32', or 'float64'.

    Example:
        >>> from keras import backend as K
        >>> inf.floatx()
        'float32'
        >>> inf.set_floatx('float16')
        >>> inf..floatx()
        'float16'

    """
    global _FLOATX
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _FLOATX = str(floatx)


def is_float(dtype):
    return dtype == "float16" or dtype == "float32" or dtype == "float64"
