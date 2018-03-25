import os
import tensorflow as tf
import inferpy as inf
import edward as ed
import numpy as np



###
from inferpy.util import tf_run_wrapper
from inferpy.util import get_total_dimension
from inferpy.util import param_to_tf
from inferpy.util import ndim

import six

import inspect

from inferpy.models import RandomVariable





######

CLASS_NAME = "class_name"
PARAMS = "params"
BASE_CLASS_NAME = "base_class_name"
NDIM = "ndim"
IS_SIMPLE="is_simple"


def __add_property(cls, name, from_dist=False):


    if from_dist:
        @tf_run_wrapper
        def param(self):
            """Distribution parameter for the mean."""
            return getattr(self.dist, name)
    else:
        def param(self):
            """Distribution parameter for the mean."""
            return getattr(self, "__"+name)


    cls.__perinstance = True
    param.__doc__ = "property for " + name
    param.__name__ = name

    setattr(cls, param.__name__, property(param))


def __add_constructor(cls, class_name, base_class_name, params, is_simple):

    def constructor(self,*args, **kwargs):

        param_dist = {}
        args_list = list(args)



        for p_name in params:
            if len(args_list) > 0:
                if p_name in kwargs:
                    raise ValueError("Wrong positional or keyword argument")

                param_dist.update({p_name : args_list[0]})
                args_list = args_list[1:]
            else:
                if kwargs.get(p_name) != None:
                    param_dist.update({p_name : kwargs.get(p_name)})




        if len(param_dist)>0:

            nd_range = {}
            d = {}
            for p,v in six.iteritems(param_dist):
                    if v != None:
                        nd_range.update({p: [0,1] if is_simple.get(p) in [None, True] else [1,2]})
                        d.update({p: 1 if is_simple.get(p) in [None, True] else get_total_dimension(v if ndim(v)==1 else v[0])})


            #check the number of dimensions
            self.__check_ndim(param_dist, nd_range)



            # get the final shape

            param_dim = 1
            if kwargs.get("dim") != None: param_dim = kwargs.get("dim")

            self_shape = (inf.replicate.get_total_size(),
                          int(np.max([get_total_dimension(v)/d.get(p)
                                  for p,v in six.iteritems(param_dist) if p != None and v!=None] +
                                 [param_dim])))


            # check that dimensions are consistent

            p_expand = [p for p, v in six.iteritems(param_dist) if p!=None and v!=None and get_total_dimension(v)>d.get(p)]
            f_expand = [get_total_dimension(param_dist.get(p))/d.get(p) for p in p_expand]

            if len([x for x in f_expand if x not in [1,self_shape[1]]])>1:
                raise ValueError("Inconsistent parameter dimensions")

            # reshape the parameters

            for p,v in six.iteritems(param_dist):
                if v != None:
                    if isinstance(v, inf.models.RandomVariable)  and self_shape==tuple(v.shape):
                        param_dist[p] = param_to_tf(v)
                    else:
                        param_dist[p] = self.__reshape_param(v, self_shape, d.get(p))

            ## Build the underliying tf object

            validate_args = kwargs.get("validate_args") if  kwargs.get("validate_args") != None else False
            allow_nan_stats = kwargs.get("allow_nan_stats") if  kwargs.get("allow_nan_stats") != None else True



            dist = getattr(ed.models, class_name)(validate_args= validate_args, allow_nan_stats=allow_nan_stats, **param_dist)

        else:
            dist = None

        observed = kwargs.get("observed") if kwargs.get("observed") != None else False
        super(self.__class__, self).__init__(dist, observed=observed)




    constructor.__doc__ = "constructor for "+class_name
    constructor.__name__ = "__init__"
    setattr(cls, constructor.__name__, constructor)

def __add_repr(cls, class_name, params):

    def repr(self):

        if self.base_object != None:
            s = ", ".join([p+"="+str(getattr(self,p)) for p in params])
            return "<inferpy "+class_name+" "+self.name+", "+s+", dtype= "+self.dist.dtype.name+" >"


    repr.__doc__ = "__repr__ for "+class_name
    repr.__name__ = "__repr__"
    setattr(cls, repr.__name__, repr)



def __check_ndim(self, params, nd_range):
    for p in [x for x in params if params.get(x)!=None]:

        n = ndim(params.get(p))
        low = nd_range.get(p)[0]
        up = nd_range.get(p)[1]

        if n < low or n > up:
            raise ValueError("Wrong parameter dimension ("+str(n)+") but should be in interval ["+str(low)+","+str(up)+"]")

def __reshape_param(self, param, self_shape, d=1):

    N = self_shape[0]
    D = self_shape[1]




    # get a D*N unidimensional vector

    k = N if get_total_dimension(param)/d == D else D*N
    param_vect = np.tile(param, k).tolist()

    ### reshape  ####
    all_num = len([x for x in param_vect if not np.isscalar(x)]) == 0

    if not all_num:
        param_vect = [param_to_tf(x) for x in param_vect]

    if N > 1:
        real_shape = [N, -1]
    else:
        real_shape = [D]
    if d > 1: real_shape = real_shape + [d]

    if all_num:
        param_np_mat = np.reshape(np.stack(param_vect), tuple(real_shape))
        param_tf_mat = tf.constant(param_np_mat, dtype="float32")
    else:
        if D == 1 and N == 1:
            param_tf_mat = param_vect[0]
        else:

            param_tf_mat = tf.reshape(tf.stack(param_vect), tuple(real_shape))



    return param_tf_mat


#####



def def_random_variable(var):




    if isinstance(var, six.string_types):
        v = {CLASS_NAME: var}
    else:
        v = var



    if not BASE_CLASS_NAME in v:
        v.update({BASE_CLASS_NAME : v.get(CLASS_NAME)})

    if not PARAMS in v:
        init_func = getattr(getattr(tf.contrib.distributions, v.get(BASE_CLASS_NAME)), "__init__")
        sig = inspect.getargspec(init_func)
        v.update({PARAMS: [x for x in sig.args if x not in ['self', 'validate_args', 'allow_nan_stats', 'name', 'dtype'] ]})


    if not IS_SIMPLE in v:
        v.update({IS_SIMPLE : {}})



    newclass = type(v.get(CLASS_NAME), (RandomVariable,),
                    {"__check_ndim" : __check_ndim,
                     "__reshape_param" : __reshape_param})



    for p in v.get(PARAMS):
        __add_property(newclass,p, from_dist=True)

    __add_constructor(newclass, v.get(CLASS_NAME), v.get(BASE_CLASS_NAME), v.get(PARAMS), v.get(IS_SIMPLE))
    __add_repr(newclass, v.get(CLASS_NAME), v.get(PARAMS))


    newclass.PARAMS = v.get(PARAMS)




    globals()[newclass.__name__] = newclass

####

class Normal(RandomVariable):
    def __init__(self, loc=0, scale=1,
                 validate_args=False,
                 allow_nan_stats=True,
                 dim=None, observed=False, name="Normal"):
        self.loc = loc
        self.scale = scale


class Beta(RandomVariable):
    def __init__(
            self,
            concentration1=None,
            concentration0=None,
            validate_args=False,
            allow_nan_stats=True,
            observed=False,
            dim=None,
            name='Beta'):
        self.concentration1 = concentration1
        self.concentration0 = concentration0



class Exponential(RandomVariable):
    def __init__(
            self,
            rate,
            validate_args=False,
            allow_nan_stats=True,
            observed = False,
            dim = None,
            name='Exponential'):
        self.rate = rate

class Uniform(RandomVariable):
    def __init__(
            self,
            low=None,
            high=None,
            validate_args=False,
            allow_nan_stats=True,
            name='Uniform',
            observed=False,
            dim=None):
        self.low = low

class Poisson(RandomVariable):
    def __init__(
            self,
            rate,
            validate_args=False,
            allow_nan_stats=True,
            name='Poisson',
            observed=False,
            dim=None):
        self.rate = rate

class Categorical(RandomVariable):
    def __init__(
            self,
            logits=None,
            probs=None,
            validate_args=False,
            allow_nan_stats=True,
            name='Categorical',
            observed=False,
            dim=None):
        self.default_logits = logits
        self.default_probs = probs

class Multinomial(RandomVariable):
    def __init__(
            self,
            total_count=None,
            logits=None,
            probs=None,
            validate_args=False,
            allow_nan_stats=True,
            name='Categorical',
            observed=False,
            dim=None):
        self.logits = logits
        self.probs = probs
        self.total_count = None

class Dirichlet(RandomVariable):
    def __init__(self,
            concentration,
            validate_args=False,
            allow_nan_stats=True,
            name='Dirichlet',
            observed=False,
            dim=None):
        self.concentration=concentration

####### run-time definition of random variables #########

ALLOWED_VARS = ["Normal", "Beta","Exponential","Uniform","Poisson"]


for v in ALLOWED_VARS:
    def_random_variable(v)



NON_SIMPLE_VARS = [{CLASS_NAME : "Categorical", IS_SIMPLE : {"probs" : False, "logits": False}},
                   {CLASS_NAME: "Multinomial", IS_SIMPLE: {"total_count":True,"probs": False, "logits": False}},
                   {CLASS_NAME: "Dirichlet", IS_SIMPLE: {"concentration": False}}
                   ]

for v in NON_SIMPLE_VARS:
    def_random_variable(v)


#####
