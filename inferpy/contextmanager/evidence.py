import contextlib
from inferpy import util


@contextlib.contextmanager
def observe(variables, data):
    sess = util.session.get_session()
    for k, v in data.items():
        if k not in variables:
            continue
        variables[k].is_observed = True
        variables[k].is_observed_var.load(True, session=sess)
        variables[k].observed_value_var.load(v, session=sess)
    try:
        yield
    finally:
        for k, v in data.items():
            if k not in variables:
                continue
            variables[k].is_observed = False
            variables[k].is_observed_var.load(False, session=sess)
