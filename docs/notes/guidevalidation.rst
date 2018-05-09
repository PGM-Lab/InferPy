Guide to Model Validation
=========================

.. note:: not implemented yet


Model validation try to assess how faifhfully the inferered
probabilistic model represents and explain the observed data.

The main tool for model validation consists on analyzing the posterior
predictive distribution,

:math:`p(y_{test}, x_{test}|y_{train}, x_{train}) = \int p(y_{test}, x_{test}|z,\theta)p(z,\theta|y_{train}, x_{train}) dzd\theta`


This posterior predictive distribution can be used to measure how well
the model fits an independent dataset using the test marginal
log-likelihood, :math:`\ln p(y_{test}, x_{test}|y_{train}, x_{train})`,

.. code:: python

    log_like = probmodel.evaluate(test_data, metrics = ['log_likelihood'])

In other cases, we may need to evalute the predictive capacity of the
model with respect to some target variable :math:`y`,

:math:`p(y_{test}|x_{test}, y_{train}, x_{train}) = \int p(y_{test}|x_{test},z,\theta)p(z,\theta|y_{train}, x_{train}) dzd\theta`

So the metrics can be computed with respect to this target variable by
using the ‘targetvar’ argument,

.. code:: python

    log_like, accuracy, mse = probmodel.evaluate(test_data, targetvar = y, metrics = ['log_likelihood', 'accuracy', 'mse'])

So, the log-likelihood metric as well as the accuracy and the mean
square error metric are computed by using the predictive posterior
:math:`p(y_{test}|x_{test}, y_{train}, x_{train})`.

Custom evaluation metrics can also be defined,

.. code:: python

    def mean_absolute_error(posterior, observations, weights=None):
        predictions = tf.map_fn(lambda x : x.getMean(), posterior)
        return tf.metrics.mean_absolute_error(observations, predictions, weights)
        
    mse, mae = probmodel.evaluate(test_data, targetvar = y, metrics = ['mse', mean_absolute_error])

