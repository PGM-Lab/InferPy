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



""" Module with the functionality for evaluating the InferPy models """



import inferpy as inf
import edward as ed
from six import iteritems




ALLOWED_METRICS = ['binary_accuracy',
'categorical_accuracy',
'sparse_categorical_accuracy',
'log_loss',
'binary_crossentropy',
'categorical_crossentropy',
'sparse_categorical_crossentropy',
'hinge',
'squared_hinge',
'mse',
'MSE',
'mean_squared_error',
'mae',
'MAE',
'mean_absolute_error',
'mape',
'MAPE',
'mean_absolute_percentage_error',
'msle',
'MSLE',
'mean_squared_logarithmic_error',
'poisson',
'cosine',
'cosine_proximity',
'log_lik',
'log_likelihood']

""" List with all the allowed metrics for evaluation """



def evaluate(metrics, data, n_samples=500, output_key=None, seed=None):

    """ Evaluate a fitted inferpy model using a set of metrics. This function
    encapsulates the equivalent Edward one.

    Args:
        metrics: list of str indicating the metrics or sccore functions to be used.


    An example of use:

    .. literalinclude:: ../../examples/evaluate.py
       :language: python
       :lines: 52,53


    Returns:
        list of float or float: A list of evaluations or a single evaluation.


    Raises:
        NotImplementedError: If an input metric does not match an implemented metric.

    """

    data_ed = {}

    for (key, value) in iteritems(data):
        data_ed.update(
            {key.dist if isinstance(key, inf.models.RandomVariable) else key :
                 value.dist if isinstance(value, inf.models.RandomVariable) else value})

    output_key_ed = output_key.dist if isinstance(output_key, inf.models.RandomVariable) else output_key

    return ed.evaluate(metrics, data_ed, n_samples, output_key_ed, seed)