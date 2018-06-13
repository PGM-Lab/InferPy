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

import edward as ed
import tensorflow as tf
from six import iteritems

import inferpy as inf


from inferpy.util import input_model_data
from inferpy.util import multishape
from inferpy.util import tf_run_wrapper


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
            if not ProbModel.compatible_var(d):
                raise ValueError("The input argument is not a list of RandomVariables")

        if ProbModel.is_active():
            raise inf.util.ScopeException("Nested probabilistic models cannot be defined")



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

    def compile(self, infMethod="KLqp", Q=None):

        """ This method initializes the structures for making inference in the model."""


        # check if the infMethod is an alias
        if infMethod in inf.INF_METHODS_ALIAS.keys():
            infMethod = inf.INF_METHODS_ALIAS.get(infMethod)


        # check if the inference method exists
        if infMethod not in inf.INF_METHODS:
            raise ValueError("Unsupported inference method: "+infMethod)

        self.infMethod = infMethod

        if Q == None:
            Q = inf.Qmodel.build_from_pmodel(self)

        self.q_vars = Q.dict

        self.propagated = False

    @input_model_data
    def fit(self, data, reset_tf_vars=True):

        """ Assings data to the observed variables"""

        if self.is_compiled()==False:
            raise Exception("Error: the model is not compiled")

        self.data = {}

        for k, v in iteritems(data):
            self.data.update({self.get_var(k).dist: v})


        self.inference = getattr(ed.inferences, self.infMethod)(self.q_vars, self.data)



        self.inference.initialize()

        sess = inf.util.Runtime.tf_sess



        if reset_tf_vars:
            tf.global_variables_initializer().run()

        else:

            for t in tf.global_variables():
                if not sess.run(tf.is_variable_initialized(t)):
                    sess.run(tf.variables_initializer([t]))


        for _ in range(self.inference.n_iter):
            info_dict = self.inference.update()
            self.inference.print_progress(info_dict)

        self.inference.finalize()



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

        ed_post = self.inference.latent_vars.get(latent_var.dist)
        vartype = type(ed_post).__name__


        if vartype in inf.models.ALLOWED_VARS:
            post = getattr(inf.models, vartype)(name="post_" + latent_var.name)
        else:
            post = inf.models.Deterministic()

        post.base_object = ed_post

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

        if not ProbModel.compatible_var(v):
            raise ValueError("The input argument must be a non-generic random variable")

        if v not in self.varlist:
            self.varlist.append(v)
            self.reset_compilation()
    @staticmethod
    def compatible_var(v):
        return (isinstance(v, inf.models.RandomVariable)
                #or isinstance(v, ed.models.RandomVariable)
                ) and not v.is_generic_variable()


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
            sd.update({v.name: tf.reshape(v.dist, shape=v.shape)})

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


    def get_parents(self, v):

        out = []
        d = self.get_vardict_rev()

        for p in v.dist.get_parents():
            if d.get(p) not in out:
                out.append(d.get(p))

        return out


    def no_parents(self):
        return [v for v in self.varlist if len(v.dist.get_parents()) == 0]

    def get_vardict(self):
        d = {}
        for v in self.varlist:
            d.update({v : v.dist})

        return d


    def get_vardict_rev(self):
        d = {}
        for v in self.varlist:
            d.update({v.dist : v})

        return d


    def copy(self, swap_dict=None):

        new_vars = {} if swap_dict == None else swap_dict

        while len(new_vars.keys()) < len(self.varlist):
            tocopy = [v for v in self.varlist
                      if (len(self.get_parents(v)) == 0 or self.get_parents(v) <= new_vars.keys())
                      and v not in new_vars.keys()]

            v = tocopy[0]

            copy = getattr(inf.models, type(v).__name__)()

            new_vars_ed = {}
            for (key, value) in iteritems(new_vars):
                new_vars_ed.update({key.dist : value.dist if isinstance(value, inf.models.RandomVariable) else value})

            copy.dist = ed.copy(v.dist, new_vars_ed)
            copy.copied_from = v

            new_vars.update({v: copy})

        return ProbModel(varlist=[v for v in new_vars.values() if isinstance(v, inf.models.RandomVariable)])


    def get_copy_from(self, original_var):
        cplst = [v for v in self.varlist if v.copied_from == original_var]

        if len(cplst) == 0:
            return None
        return cplst[0]



    def predict(self, target, data, reset_tf_vars=False):

        # check learnt

        local_hidden = [z for z in target.get_local_hidden() if z not in data.keys()]
        global_hidden = [h for h in self.latent_vars if h not in local_hidden and h not in data.keys()]
        other_observed = [a for a in self.observed_vars if a not in data.keys() and a != target]


        # add posterior of the latent variables
        for h in global_hidden:
            if h not in data.keys():
                data.update({h: self.posterior(h)})

        data_ed = {}
        for (key, value) in iteritems(data):
            data_ed.update(
                {key.dist if isinstance(key, inf.models.RandomVariable) else key :
                     value.dist if isinstance(value, inf.models.RandomVariable) else value})






        q_target = inf.Qmodel.new_qvar(target, check_observed=False)

        latent_vars_ed = {target.dist : q_target.dist}

        for z in local_hidden:
            qz = inf.Qmodel.new_qvar(z, check_observed=False)
            latent_vars_ed.update({z.dist : qz.dist})

        for a in other_observed:
            qa = inf.Qmodel.new_qvar(a, check_observed=False)
            latent_vars_ed.update({a.dist : qa.dist})



        inference_pred = ed.ReparameterizationKLqp(latent_vars_ed, data=data_ed)
        #inference_pred.run()
        inference_pred.initialize()

        sess = inf.util.Runtime.tf_sess

        if reset_tf_vars:
            tf.global_variables_initializer().run()
        else:
            for t in tf.global_variables():
                if not sess.run(tf.is_variable_initialized(t)):
                    sess.run(tf.variables_initializer([t]))


        for _ in range(inference_pred.n_iter):
            info_dict = inference_pred.update()
            inference_pred.print_progress(info_dict)

        inference_pred.finalize()



#        tf.graph_util.convert_variables_to_constants(inf.util.Runtime.tf_sess, tf.get_default_graph())

        return q_target

    def predict_old(self, target_var, observations):

        ### copy qvars from the model

        # check learnt

        # add posterior of the latent variables
        for h in self.latent_vars:
            if h not in observations.keys():
                observations.update({h: self.posterior(h)})

        ancestors = target_var.dist.get_ancestors()
        ancestors_obs = dict([(obs, observations[obs]) for obs in observations.keys() if obs.dist in ancestors])

        m_pred = self.copy(swap_dict=ancestors_obs)

        non_ancestors_obs = dict([(m_pred.get_copy_from(obs), observations[obs]) for obs in observations.keys() if
                                  obs.dist not in ancestors])

        non_ancestors_obs_ed = {}
        for (key, value) in iteritems(non_ancestors_obs):
            non_ancestors_obs_ed.update(
                {key.dist: value.dist if isinstance(value, inf.models.RandomVariable) else value})

        copy_target = m_pred.get_copy_from(target_var)
        q_target = inf.Qmodel.new_qvar(copy_target, check_observed=False)

        inference_pred = ed.KLqp({copy_target.dist: q_target.dist},
                                 data=non_ancestors_obs_ed)

        copy_target.dist.get_parents()

        inference_pred.run()

        return q_target