import contextlib
import numpy as np
from inferpy import util


@contextlib.contextmanager
def observe(variables, data):
    # default session
    sess = util.session.get_session()
    # iterate for all variables both in `variables` and `data`
    for k, v in data.items():
        if k not in variables:
            continue
        # Set the variable to observed. This `tf.Variable` is used by the interceptor `set_values_condition`
        variables[k].is_observed.load(True, session=sess)
        # Now load the value into the `tf.Variable`:
        # if has shape attr:
        if hasattr(v, 'shape'):
            # shape of tf.Variable and value matches
            if v.shape == variables[k].observed_value.shape:
                variables[k].observed_value.load(v, session=sess)
            # shape of tf.Variable and value without the sample_shape (first dim) matches
            # NOTE: this might happend if data comes from sample() and sample_shape == 1
            elif len(v.shape) > 0 and v.shape[0] == 1 and v.shape[1:] == variables[k].observed_value.shape:
                variables[k].observed_value.load(v[0], session=sess)
            # otherwise, just try to do broadcast and load the value
            else:
                # try to broadcast v using numpy (it cannot be a tensor)
                variables[k].observed_value.load(
                    np.broadcast_to(v, variables[k].observed_value.shape.as_list()), session=sess)
        else:
            # try to broadcast v using numpy (it cannot be a tensor)
            variables[k].observed_value.load(
                np.broadcast_to(v, variables[k].observed_value.shape.as_list()), session=sess)
    try:
        yield
    finally:
        # just needs to revert the `is_observed` tf.Variable
        for k, v in data.items():
            if k not in variables:
                continue
            variables[k].is_observed.load(False, session=sess)
