Guide to Bayesian Deep Learning
===============================

InferPy inherits Edward's approach for representing probabilistic models
as (stochastic) computational graphs. As describe above, a random
variable :math:`x` is associated to a tensor :math:`x^*` in the
computational graph handled by TensorFlow, where the computations takes
place. This tensor :math:`x^*` contains the samples of the random
variable :math:`x`, i.e. :math:`x^* \sim p(x|\theta)`. In this way,
random variables can be involved in complex deterministic operations
containing deep neural networks, math operations and another libraries
compatible with Tensorflow (such as Keras).

Bayesian deep learning or deep probabilistic programming enbraces the
idea of employing deep neural networks within a probabilistic model in
order to capture complex non-linear dependencies between variables.

InferPy's API gives support to this powerful and flexible modeling
framework. Let us start by showing how a non-linear PCA can be defined by mixing ``tf.layers`` and InferPy code.

.. literalinclude:: ../../examples/docs/guidebayesian/1.py
   :language: python3
   :lines: 9-62





In this case, the parameters of the decoder neural network (i.e., weights)
are automatically managed by TensorFlow. These parameters are them treated as
model parameters and not exposed to the user. In consequence, we can not
be Bayesian about them by defining specific prior distributions.


Alternatively, we could use Keras layers by simply defining an alternative
decoder function as follows.


.. literalinclude:: ../../examples/docs/guidebayesian/1.py
   :language: python3
   :lines: 68-75

