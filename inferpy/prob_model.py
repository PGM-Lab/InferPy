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


"""Module with the probabilistic model functionality.
"""

import inferpy.util
import inferpy.models
from inferpy.util import tf_run_wrapper
from inferpy.util import multishape
from inferpy.util import input_model_data
import tensorflow as tf
import edward as ed
import numpy as np

from six import iteritems
from functools import wraps

from inferpy.qmodel import Qmodel


import pandas as pd


class ProbModel(object):
    """Class implementing a probabilistic model

    A probabilistic model defines a joint distribution over observed and latent variables. This
    class encapsulates all the functionality for making inference (and learning) in these models.

    An example of use:

    .. literalinclude:: ../../examples/prob_model_def.py


    This class can be used, for instance, for infering the parameters of some observed data:

    .. literalinclude:: ../../examples/simple_inference_params.py



    """



    __active_models = []
    """ list: static variable that contains the models defined by means
    of the construct 'with' that are active"""


    def __init__(self, varlist=None):
        """Initializes a probabilistic model

        Args:
            varlist: optional list with the variables in the model

        """


        if varlist==None:
            self.varlist = []
        else:
            self.varlist=varlist


        for d in self.varlist:
            if not self.compatible_var(d):
                raise ValueError("The input argument is not a list of RandomVariables")

        if ProbModel.is_active():
            raise inferpy.util.ScopeException("Nested probabilistic models cannot be defined")



        self.q_vars = {}
        self.data = {}

        self.propagated = False

    # properties and setters

    @property
    def varlist(self):
        """ list of variables (observed and latent)"""
        return self.__varlist

    @varlist.setter
    def varlist(self,varlist):

        """ modifies the list of variables in the model"""

        self.__varlist = varlist
        self.reset_compilation()

    @property
    def observed_vars(self):

        """ list of observed variabels in the model"""

        vl = []
        for v in self.varlist:
            if v.observed:
                vl.append(v)
        return vl


    @property
    def latent_vars(self):

        """ list of latent (i.e., non-observed) variables in the model """

        vl = []
        for v in self.varlist:
            if v.observed==False:
                vl.append(v)
        return vl

    # other methods

    def compile(self):

        """ This method initializes the structures for making inference in the model."""

        self.q_vars = {}
        for v in self.latent_vars:
            #qv = ed.models.Normal(loc=tf.Variable(np.zeros(v.dim), dtype="float32"),
            #                      scale=tf.Variable(np.ones(v.dim), dtype="float32"),
            #                      name = "q_"+str.replace(v.name, ":", ""))

            qv = Qmodel.generate_ed_qvar(v)
            self.q_vars.update({v.base_object: qv})


        self.propagated = False

    @input_model_data
    def fit(self, data):

        """ Assings data to the observed variables"""

        if self.is_compiled()==False:
            raise Exception("Error: the model is not compiled")

        self.data = {}

        for k, v in iteritems(data):
            self.data.update({self.get_var(k).dist: v})

        self.q_vars.get(self.latent_vars[0].dist)



        self.inference = ed.KLqp(self.q_vars, self.data)
        self.inference.run()
        self.propagated = True

    @multishape
    def posterior(self, latent_var):

        """ Return the posterior distribution of some latent variables

            Args:
                latent_var: a single or a set of latent variables in the model

            Return:
                Random variable(s) of the same type than the prior distributions

        """

        if self.propagated == False:
            self.inference.run()
            self.propagated = True

        post = getattr(inferpy.models, type(latent_var).__name__)(name="post_"+latent_var.name)
        post.dist = self.inference.latent_vars.get(latent_var.dist)

        return post


    def __enter__(self):
        """ Method for allowing the use of the construct 'with' """
        ProbModel.__active_models.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Method for allowing the use of the construct 'with' """
        ProbModel.__active_models.pop()

    def add_var(self, v):
        """ Method for adding a new random variable. After use, the model should be re-compiled """

        if not self.compatible_var(v):
            raise ValueError("The input argument must be a non-generic random variable")

        if v not in self.varlist:
            self.varlist.append(v)
            self.reset_compilation()

    def compatible_var(self, v):
        return isinstance(v, inferpy.models.RandomVariable) and not v.is_generic_variable()


    def get_var(self,name):

        """ Get a varible in the model with a given name """

        for v in self.varlist:
            if v.name ==  name:
                return v
        return None



    @tf_run_wrapper
    def log_prob(self, sample_dict):

        """ Computes the log probabilities of a (set of) sample(s)"""

        sd = {}

        for k, v in iteritems(sample_dict):
            var=self.get_var(k)
            sd.update({k: var.log_prob(v, tf_run=False)})


        return sd

    @tf_run_wrapper
    def sum_log_prob(self, sample_dict):

        """ Computes the sum of the log probabilities of a (set of) sample(s)"""

        lp=0

        for k, v in iteritems(sample_dict):
            var = self.get_var(k)
            lp += var.sum_log_prob(v, tf_run=False)

        return lp

    @tf_run_wrapper
    def sample(self, size=1):
        """ Generates a sample for eache variable in the model """
        sd = {}
        for v in self.varlist:
            sd.update({v.name:v.sample(size, tf_run=False)})

        return sd

    def reset_compilation(self):

        """ Clear the structues created during the compilation of the model """

        self.q_vars = {}
        self.data = {}
        self.propagated = False

    def is_compiled(self):
        """ Determines if the model has been compiled """
        return len(self.q_vars) > 0

    def get_config(self):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


    # static methods
    @staticmethod
    def get_active_model():

        """ Return the active model defined with the construct 'with' """

        if ProbModel.is_active():
            return ProbModel.__active_models[-1]
        return []

    @staticmethod
    def is_active():
        """Check if a replicate construct has been initialized

        Returns:
             True if the method is inside a construct ProbModel (of size different to 1).
             Otherwise False is return
        """
        return len(ProbModel.__active_models)>0


