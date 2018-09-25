# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Module with the required functionality for implementing the Q-models and Q-distributions used by
some inference algorithms.
"""


import inferpy as inf
import tensorflow as tf
import edward as ed
import sys




class Qmodel(object):

    """Class implementing a Q model

    A Q model approximates the posterior distribution of a probabilistic model P .
    In the ‘Q’ model we should include a q distribution for every non observed variable in the ‘P’ model.
    Otherwise, an error will be raised during model compilation.

    An example of use:

    .. literalinclude:: ../../examples/q_model_inference.py


    """



    def __init__(self, varlist):

        self.varlist = varlist

    """Initializes a Q model

    Args:
        varlist: list with the variables in the model

    """

    @staticmethod
    def build_from_pmodel(p, empirical=False):
        """Initializes a Q model from a P model.

        Args:
            p: P model of type inferpy.ProbModel.
            empirical: determines if q distributions will be empirical or of the same type than each p distribution.

        """
        return inf.Qmodel([inf.Qmodel.new_qvar(v) if not empirical else inf.Qmodel.new_qvar_empirical(v,1000) for v in p.latent_vars])



    @property
    def dict(self):
        """ Dictionary where the keys and values are the p and q distributions respectively. """


        d = {}
        for v in self.varlist:
            d.update({v.bind.dist : v.dist})
        return d


    @property
    def varlist(self):
        """ list of q variables of Inferpy type"""
        return self.__varlist

    @varlist.setter
    def varlist(self,varlist):
        """ modifies the list of variables in the model"""
        for v in varlist:
            if not Qmodel.compatible_var(v):
                raise ValueError("Non-compatible var "+v.name)

        self.__varlist = varlist


    def add_var(self, v):
        """ Method for adding a new q variable """

        if not Qmodel.compatible_var(v):
            raise ValueError("The input argument must be a q variable")

        if v not in self.varlist:
            self.varlist.append(v)



    @staticmethod
    def compatible_var(v):
        return inf.ProbModel.compatible_var(v) and v.bind != None



    @staticmethod
    def __generate_ed_qvar(v, initializer, vartype, params):

        """ Builds an Edward q-variable for a p-variable.

        Args:
            v: Edward variable to be approximated.
            initializer (str): indicates how the new variable should be initialized. Possible values: "ones" , "zeroes".
            vartype (str): Edward type of the new variable.
            params: lists of strings indicating the paramaters of the new variable.


        Returns:
            Edward variable approximating in input variable


        """

        qparams = {}

        if initializer == "ones":
            init_f = tf.ones
        elif initializer == "zeroes":
            init_f = tf.zeros
        else:
            raise ValueError("Unsupported initializer: "+initializer)



        for p_name in params:
            if p_name not in ["probs"]:

                p_shape = getattr(v, p_name).shape.as_list() if hasattr(v, p_name) else v.shape


                var = tf.Variable(init_f(p_shape), dtype="float32", name="q_"+v.name+p_name)
                inf.util.Runtime.tf_sess.run(tf.variables_initializer([var]))


                var = tf.where(tf.is_nan(var), tf.zeros_like(var), var)

                if p_name in ["scale"]:
                    var = tf.nn.softplus(var)
                elif p_name in ["probs"]:
                    var = tf.clip_by_value(tf.where(tf.is_nan(var), tf.zeros_like(var), var),1e-8, 1)
                elif p_name in ["logits"]:
                    var = tf.where(tf.is_nan(var), tf.zeros_like(var), var)

                qparams.update({p_name: var})


        qvar =  getattr(ed.models, vartype)(name = "q_"+str.replace(v.name, ":", ""),allow_nan_stats=False, **qparams)

        return qvar



    @staticmethod
    def new_qvar(v, initializer='ones', qvar_inf_module=None, qvar_inf_type = None, qvar_ed_type = None, check_observed = True, name="qvar"):

        """ Builds an Inferpy q-variable for a p-variable.

        Args:
            v: Inferpy variable to be approximated.
            initializer (str): indicates how the new variable should be initialized. Possible values: "ones" , "zeroes".
            qvar_inf_module (str): module of the new Inferpy variable.
            qvar_inf_type (str): name of the new Inferpy variable.
            qvar_ed_type (str): full name of the encapsulated Edward variable type.
            check_observed (bool): To check if p-variable is observed.

        Returns:
            Inferpy variable approximating in input variable.


        """


        if not inf.ProbModel.compatible_var(v):
            raise ValueError("Non-compatible variable")
        elif v.observed and check_observed:
            raise ValueError("Variable "+v.name+" cannot be observed")

        ## default values ##

        if qvar_inf_module == None:
            qvar_inf_module = sys.modules[v.__class__.__module__]

        if qvar_inf_type == None:
            qvar_inf_type = type(v).__name__

        if qvar_ed_type == None:
            qvar_ed_type = type(v).__name__



        qv = getattr(qvar_inf_module, qvar_inf_type)()
        qv.dist = Qmodel.__generate_ed_qvar(v.dist, initializer, qvar_ed_type, v.PARAMS)
        qv.bind = v
        return qv




    @staticmethod
    def new_qvar_empirical(v, n_post_samples, initializer='ones', check_observed = True, name="qvar"):

        """ Builds an empirical Inferpy q-variable for a p-variable.

        Args:
            v: Inferpy variable to be approximated.
            n_post_samples: number of posterior samples.
            initializer (str): indicates how the new variable should be initialized. Possible values: "ones" , "zeroes".
            check_observed (bool): To check if p-variable is observed.

        Returns:
            Inferpy variable approximating in input variable. The InferPy type will be Deterministic while
            the encapsulated Edward variable will be of type Empirical.


        """


        if not inf.ProbModel.compatible_var(v):
            raise ValueError("Non-compatible variable")
        elif v.observed and check_observed:
            raise ValueError("Variable "+v.name+" cannot be observed")




        qv = inf.models.Deterministic()


        if initializer == "ones":
            init_f = tf.ones
        elif initializer == "zeroes":
            init_f = tf.zeros
        else:
            raise ValueError("Unsupported initializer: "+initializer)


        dtype = v.base_object.dtype.name
        var = tf.Variable(init_f([n_post_samples] + v.shape, dtype=dtype), dtype=dtype, name="q_"+v.name + "/params")
        qv.base_object = ed.models.Empirical(params=var, name = "q_"+str.replace(v.name, ":", ""))


        qv.bind = v
        return qv



    @staticmethod
    def Empirical(v, n_post_samples=500, initializer='ones'):
        return inf.Qmodel.new_qvar_empirical(v, n_post_samples, initializer)


####





def __add__new_qvar(vartype):

    name = vartype.__name__

    def f(cls,v, initializer='ones'):
        return cls.new_qvar(v, initializer, qvar_inf_type=name)


    f.__doc__ = "Creates a new q-variable of type " + name + "" \
                                                             "" \
                                                             "\n\n        Args:" \
                                                             "\n            v: Inferpy p-variable" \
                                                             "\n            initializer (str): indicates how the new variable " \
                                                             "should be initialized. Possible values: 'ones' , 'zeroes'." \
                                                             "";
    f.__name__ = name

    setattr(Qmodel, f.__name__, classmethod(f))


for vartype in inf.models.RandomVariable.__subclasses__():
    __add__new_qvar(vartype)


