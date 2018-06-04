from functools import wraps
import inferpy as inf
import numpy as np
from six import iteritems

import pandas as pd



def tf_run_wrapper(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        if "tf_run" in kwargs:
            tf_run = kwargs.pop("tf_run")
        else:
            tf_run = inf.util.Runtime.tf_run_default


        if tf_run:

            # transforms in a list
            output_tf = f(*args, **kwargs)
            if type(output_tf).__module__ == np.__name__ or type(output_tf).__name__ == list.__name__:
                output_tf_vect = output_tf
            elif type(output_tf).__name__ == dict.__name__:
                output_tf_vect = list(output_tf.values())
            else:
                output_tf_vect = [output_tf]

            # evaluation
            output_eval_vect = inf.util.Runtime.tf_sess.run(output_tf_vect)


            # transforms in original type
            if type(output_tf).__module__ == np.__name__ or type(output_tf).__name__ == list.__name__:
                output_eval = output_eval_vect
            elif type(output_tf).__name__ == dict.__name__:
                output_eval = {}
                i = 0
                for k, v in iteritems(output_tf):
                    output_eval.update({k: output_eval_vect[i]})
                    i = i+1
            else:
                output_eval = output_eval_vect[0]


            ###
            #output_tf = f(*args, **kwargs)
            #if type(output_tf).__module__ == np.__name__:
            #    output_eval = np.array([])
            #    for t in output_tf:
            #        output_eval.append(inf.util.Runtime.tf_sess.run(t))
            #elif type(output_tf).__name__ == dict.__name__:
            #    output_eval = {}
            #    for k, v in iteritems(output_tf):
            #        output_eval.update({k:inf.util.Runtime.tf_sess.run(v)})
            #else:
            #    output_eval = inf.util.Runtime.tf_sess.run(f(*args, **kwargs))

            return output_eval
        return f(*args, **kwargs)
    return wrapper



def multishape(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        first_arg = 1
#        if args[first_arg].__class__ in [[].__class__, {}.__class__, np.array([]).__class__]:            # single element

        if np.ndim(args[first_arg]) == 0:            # single element
            return f(*args, **kwargs)
        elif np.ndim(args[first_arg]) == 1:        # unidimensional vector
            output = []
            for i in args[1]:

                if first_arg == 1:
                    output.append(f(args[0], i, **kwargs))
                else:
                    output.append(f(i, **kwargs))
            return output
        else:
            raise ValueError("@multishape wrapper can only be applied to single elements or to 1-dimension vectors")


    return wrapper



def static_multishape(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        first_arg = 0
#        if args[first_arg].__class__ in [[].__class__, {}.__class__, np.array([]).__class__]:            # single element

        if np.ndim(args[first_arg]) == 0:            # single element
            return f(*args, **kwargs)
        elif np.ndim(args[first_arg]) == 1:        # unidimensional vector
            output = []
            for i in args[first_arg]:

                if first_arg == 1:
                    output.append(f(args[0], i, **kwargs))
                else:
                    output.append(f(i, **kwargs))
            return output
        else:
            raise ValueError("@multishape wrapper can only be applied to single elements or to 1-dimension vectors")


    return wrapper





def singleton(class_):
    class class_w(class_):
        _instance = None
        def __new__(class_, *args, **kwargs):
            if class_w._instance is None:
                class_w._instance = super(class_w,
                                    class_).__new__(class_,
                                                    *args,
                                                    **kwargs)
                class_w._instance._sealed = False
            return class_w._instance
        def __init__(self, *args, **kwargs):
            if self._sealed:
                return
            super(class_w, self).__init__(*args, **kwargs)
            self._sealed = True
    class_w.__name__ = class_.__name__
    return class_w

def input_model_data(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        self = args[0]
        data = args[1]

        if isinstance(data, pd.DataFrame):
            newdata =  {}

            for k, v in iteritems(data.to_dict(orient="list")):
                if self.get_var(k) != None:

                    newdata.update({k : np.reshape(v, (np.size(v),1))})


        elif isinstance(data, dict):
            newdata = data
        else:
            raise ValueError("Wrong input data type: it should be a pandas dataframe or a dictionary")


        return f(self,newdata)

    return wrapper