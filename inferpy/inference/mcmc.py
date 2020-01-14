# https://github.com/PGM-Lab/BBVI-TFP/blob/e45b1d654edb0f014665b719fdfc461429832f50/playground/edward2/log-regression-MCMC.py

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import edward2 as ed

from .inference import Inference
from inferpy import models
from inferpy.queries import Query
from inferpy import util
from inferpy.data.loaders import build_sample_dict


class MCMC(Inference):
    def __init__(self, step_size=0.01, num_leapfrog_steps=5, num_burnin_steps=1000, num_results=500):
        """Creates a new Markov Chain MonteCarlo (MCMC) Inference object.
            Args:
                step_size: Tensor or Python list of Tensors representing the step size for the leapfrog integrator.
                           Must broadcast with the shape of current_state. Larger step sizes lead to faster progress,
                           but too-large step sizes make rejection exponentially more likely. When possible, it's often
                           helpful to match per-variable step sizes to the standard deviations of the target distribution
                           in each variable.
                num_leapfrog_steps: Integer number of steps to run the leapfrog integrator for. Total progress per HMC
                                    step is roughly proportional to step_size * num_leapfrog_steps.
                num_burnin_steps: Integer number of chain steps to take before starting to collect results.
                                  Default value: 0 (i.e., no burn-in).
                num_results: Integer number of Markov chain draws.
        """

        self.step_size = step_size

        self.num_leapfrog_steps = num_leapfrog_steps

        self.num_burnin_steps = num_burnin_steps

        self.num_results = num_results

        # pmodel not established yet
        self.pmodel = None
        # The size of the plate when expand the models
        self.plate_size = None

        # tensors where the results of applying MCMC are stored
        self._states_tensor = None
        self._kernel_results_tensor = None
        # the final samples computed by applying the method
        self.states = None

        # expanded variables and parameters
        self.expanded_variables = None
        self.expanded_parameters = None

        # not observed vars
        self.hiddenvars_name = None

    def compile(self, pmodel, data_size, extra_loss_tensor=None):
        # set the used pmodel
        self.pmodel = pmodel
        # and the plate size, which matches the data size
        self.plate_size = data_size
        # extra_loss_tensor comes from inf.layers.Sequential losses, which cannot be used with this inference method
        if extra_loss_tensor is not None:
            raise RuntimeError("The MCMC inference method cannot be used with models containing layers from tf, keras or inferpy.")

    def update(self, data):
        # data must be a sample dictionary
        sample_dict = build_sample_dict(data)
        # ensure that the size of the data matches with the self.plate_size
        data_size = util.iterables.get_plate_size(self.pmodel.vars, sample_dict)
        if data_size != self.plate_size:
            raise ValueError("The size of the data must be equal to the plate size: {}".format(self.plate_size))

        sess = util.get_session()

        with util.interceptor.disallow_conditions():
            with ed.interception(util.interceptor.set_values(**sample_dict)):
                # create the hmc kernel
                self._generate_sample_chain(sample_dict)

                variables_states, _ = sess.run([self._states_tensor, self._kernel_results_tensor])

        # event_ndims is the number of dims of states minus 1 because of the dimension of number os samples
        self.states = {name: models.Empirical(states, event_ndims=len(states.shape) - 1, name=name)
                       for name, states in zip(self.hiddenvars_name, variables_states)}

    def posterior(self, target_names=None, data={}):
        return Query(self.states, target_names, data)

    def posterior_predictive(self, target_names=None, data={}):
        # posterior_predictive uses pmodel variables, but global hidden (parameters) intercepted with qmodel variables.
        expanded_data = {
            **data,
            **(util.runtime.try_run({k: v.sample() for k, v in self.states.items() if k not in data}))
        }
        return Query(self.pmodel.vars, target_names, expanded_data)

    ########################
    # Auxiliar functions
    ########################

    def _generate_sample_chain(self, data):

        # check if model should be expanded for getting the the initial state
        local_hidden = [n for n, v in self.pmodel.vars.items() if v.is_datamodel and n not in data.keys()]
        if len(local_hidden)>0:
            init_vars, _ = self.pmodel.expand_model(self.plate_size)
        else:
            init_vars = self.pmodel.vars

        # sample the initial state
        self.hiddenvars_name = []
        initial_state = []
        for name, var in init_vars.items():
            if name not in data:
                # sample vars to use them as initial state
                initial_state.append(var)
                self.hiddenvars_name.append(name)
        initial_state = util.get_session().run(initial_state)

        # initialize MCMC
        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self._target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps
        )

        self._states_tensor, self._kernel_results_tensor = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            current_state=initial_state,
            kernel=hmc_kernel,
            num_burnin_steps=self.num_burnin_steps
        )

    def _target_log_prob_fn(self, *hiddenvars_tensors):
        # expand de pmodel, using the intercept.set_values function, to include the sample_dict (done in `update`)
        # and the hiddenvars_tensors
        with ed.interception(util.interceptor.set_values(**{k: v for k, v in zip(
                self.hiddenvars_name, hiddenvars_tensors
                )})):
            self.expanded_variables, self.expanded_parameters = self.pmodel.expand_model(self.plate_size)

        energy = tf.reduce_sum(
            [tf.reduce_sum(p.log_prob(p.value)) for p in self.expanded_variables.values()])

        return energy
