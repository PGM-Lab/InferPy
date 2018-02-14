from functools import wraps
import inferpy as inf

def tf_run_wrapper(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        if "tf_run" in kwargs:
            tf_run = kwargs.pop("tf_run")
        else:
            tf_run = True


        if tf_run:
            return inf.util.runtime.tf_sess.run(f(*args, **kwargs))
        return f(*args, **kwargs)
    return wrapper


## alternative code not working in python 2.x

#def tf_run_wrapper(f):
#    @wraps(f)
#    def wrapper(*args, tf_run=True, **kwargs):
#        if tf_run:
#            return inf.util.runtime.tf_sess.run(f(*args, **kwargs))
#        return f(*args, **kwargs)
#    return wrapper


