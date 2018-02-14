from functools import wraps
import inferpy as inf


def tf_run_wrapper(f):
    @wraps(f)
    def wrapper(*args, tf_run=True, **kwds):
        if tf_run:
            return inf.util.runtime.tf_sess.run(f(*args, **kwds))
        return f(*args, **kwds)
    return wrapper