import numpy as np
import warnings

from inferpy import contextmanager
from inferpy import util


class Query:
    def __init__(self, variables, target_names=None, data={}):
        self.target_variables = variables if not target_names else \
            {k: v for k, v in variables.items() if k in target_names}
        # warn if target variables contains observations in data (will be removed)
        prev_len = len(self.target_variables)
        self.target_variables = {k: v for k, v in self.target_variables.items() if k not in data}
        if len(self.target_variables) != prev_len:
            warnings.warn("Some target variables has been removed because targets cannot be observations.")

        self.observed_variables = variables
        self.data = data

    def log_prob(self):
        """ Computes the log probabilities of a (set of) sample(s)"""
        with contextmanager.observe(self.observed_variables, self.data):
            return util.runtime.try_run({k: v.log_prob(v.value) for k, v in self.target_variables.items()})

    def sum_log_prob(self):
        """ Computes the sum of the log probabilities (evaluated) of a (set of) sample(s)"""
        return np.sum([np.mean(lp) for lp in self.log_prob().values()])

    def sample(self):
        """ Generates a sample for eache variable in the model """
        with contextmanager.observe(self.observed_variables, self.data):
            return util.runtime.try_run({k: (v.sample(v.sample_shape) if v.sample_shape else v.sample())
                                        for k, v in self.target_variables.items()})

    def parameters(self, names=None):
        """ Return the parameters of the Random Variables of the model.
        If `names` is None, then return all the parameters of all the Random Variables.
        If `names` is a list, then return the parameters specified in the list (if exists) for all the Random Variables.
        If `names` is a dict, then return all the parameters specified (value) for each Random Variable (key).

        NOTE: If tf_run=True, but any of the returned parameters is not a Tensor *and therefore cannot be evaluated)
            this returns a not evaluated dict (because the evaluation will raise an Exception)

        Args:
            names: A list, a dict or None. Specify the parameters for the Random Variables to be obtained.

        Returns:
            A dict, where the keys are the names of the Random Variables and the values a dict of parameters (name-value)
        """
        # argument type checking
        if not(names is None or isinstance(names, (list, dict))):
            raise TypeError("The argument 'names' must be None, a list or a dict, not {}.".format(type(names)))
        # now we can assume that names is None, a list or a dict

        # function to filter the parameters for each Random Variable
        def filter_parameters(varname, parameters):
            parameter_names = list(parameters.keys())
            if names is None:
                # use all the parameters
                selected_parameters = parameter_names
            else:
                # filter by names; if is a dict and key not in, use all the parameters
                selected_parameters = set(names if isinstance(names, list) else names.get(varname, parameters))

            return {k: util.runtime.try_run(v) for k, v in parameters.items() if k in selected_parameters}

        with contextmanager.observe(self.observed_variables, self.data):
            return {k: filter_parameters(k, v.parameters) for k, v in self.target_variables.items()}
