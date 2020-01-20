Guide to Bayesian Deep Learning
===============================


Models Containing Neural Networks
-----------------------------------

InferPy inherits Edward's approach for representing probabilistic models
as (stochastic) computational graphs. As described above, a random
variable :math:`x` is associated to a tensor :math:`x^*` in the
computational graph handled by TensorFlow, where the computations take
place. This tensor :math:`x^*` contains the samples of the random
variable :math:`x`, i.e. :math:`x^* \sim p(x|\theta)`. In this way,
random variables can be involved in complex deterministic operations
containing deep neural networks, math operations and other libraries
compatible with TensorFlow (such as Keras).

Bayesian deep learning or deep probabilistic programming embraces the
idea of employing deep neural networks within a probabilistic model in
order to capture complex non-linear dependencies between variables.
This can be done by combining InferPy with ``tf.layers``, ``tf.keras`` or ``tfp.layers``.

InferPy's API gives support to this powerful and flexible modeling
framework. Let us start by showing how a non-linear PCA.

.. literalinclude:: ../../examples/docs/guidebayesian/1.py
   :language: python3
   :lines: 9-62



In this case, the parameters of the decoder neural network (i.e., weights)
are automatically managed by TensorFlow. These parameters are treated as
model parameters and not exposed to the user. In consequence, we can not
be Bayesian about them by defining specific prior distributions.


Alternatively, we could use Keras layers by simply defining an alternative
decoder function as follows.


.. literalinclude:: ../../examples/docs/guidebayesian/1.py
   :language: python3
   :lines: 68-75

InferPy is also compatible with Keras models such as `tf.keras.Sequential``:

.. literalinclude:: ../../examples/docs/guidebayesian/1.py
   :language: python3
   :lines: 83-91


Bayesian Neural Networks
-----------------------------------

InferPy allows the definition of Bayesian NN using the same dense variational layers
that are available in ``tfp.layers``, i.e.:

- DenseFlipout: Densely-connected layer class with Flipout estimator.

- DenseLocalReparameterization: Densely-connected layer class with local reparameterization estimator.

- DenseReparameterization: Densely-connected layer class with reparameterization estimator.


The weights of these layers are drawn from distributions whose posteriors are calculated
using variational inference. For more details, check the official `tfp documentation <https://www.tensorflow.org/probability/api_docs/python/tfp/layers/dense_variational>`_.
For its usage, we simply need to include them in an InferPy
Sequential model ``inf.layers.Sequential`` as follows.


.. literalinclude:: ../../examples/docs/guidebayesian/1.py
   :language: python3
   :lines: 97-110

Note that this model differs from the one provided by Keras. A more detailed example
with Bayesian layers is given `here <../notes/bayesianNN.html>`_.


