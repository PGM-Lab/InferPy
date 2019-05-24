import contextlib
import numpy as np
from inferpy import util


@contextlib.contextmanager
def observe(variables, data):
    sess = util.session.get_session()
    for k, v in data.items():
        if k not in variables:
            continue
        variables[k].is_observed = True
        variables[k].is_observed_var.load(True, session=sess)
        if hasattr(v, 'shape'):
            if v.shape == variables[k].observed_value_var.shape:
                variables[k].observed_value_var.load(v, session=sess)
            elif len(v.shape) > 0 and v.shape[1:] == variables[k].observed_value_var.shape:
                variables[k].observed_value_var.load(v[0], session=sess)
            else:
                # try to broadcast v using numpy (it cannot be a tensor)
                variables[k].observed_value_var.load(
                    np.broadcast_to(v, variables[k].observed_value_var.shape.as_list()), session=sess)
        else:
            # try to broadcast v using numpy (it cannot be a tensor)
            variables[k].observed_value_var.load(
                np.broadcast_to(v, variables[k].observed_value_var.shape.as_list()), session=sess)
    try:
        yield
    finally:
        for k, v in data.items():
            if k not in variables:
                continue
            variables[k].is_observed = False
            variables[k].is_observed_var.load(False, session=sess)
