Guide to Approximate Inference
==============================

Getting Started with Approximate Inference
------------------------------------------

The API defines the set of algorithms and methods used to perform
inference in a probabilistic model :math:`p(x,z,\theta)` (where
:math:`x` are the observations, :math:`z` the local hidden variables,
and :math:`\theta` the global parameters of the model). More precisely,
the inference problem reduces to compute the posterior probability over
the latent variables given a data sample
:math:`p(z,\theta | x_{train})`, because by looking at these
posteriors we can uncover the hidden structure in the data. For the
running example, the posterior over the local hidden variables :math:`p(w_n|x_{train})`
tell us the latent vector representation of the sample :math:`x_n`, while the posterior
over the global variables :math:`p(\mu|x_{train})` tells us which is the affine transformation
between the latent space and the observable space. 

.. where the centroids of the data are, while :math:`p(z_n|x_{train})` shows us to which centroid every data point belongs to.

InferPy inherits Edward's approach an consider approximate inference
solutions,

.. math::  q(z,\theta) \approx p(z,\theta | x_{train})



in which the task is to approximate the posterior
:math:`p(z,\theta | x_{train})` using a family of distributions,
:math:`q(z,\theta; \lambda)`, indexed by a parameter vector
:math:`\lambda`.



Following InferPy guiding principles, users can further configure the
inference algorithm. First, they can define a model 'Q' for approximating the 
posterior distribution,


.. literalinclude:: ../../examples/docs/guideinference/1.py
   :language: python
   :lines: 36-43




In the 'Q' model we should include a q distribution for every non observed variable in 
the 'P' model. Otherwise, an error will be raised during model compilation. 

By default, the posterior **q** belongs to the same distribution family
than **p** , but in the above example we show how we can change that
(e.g. we set the posterior over **mu** to obtain a point mass estimate
instead of the Gaussian approximation used by default). We can also
configure how these **q's** are initialized using any of the Keras's
initializers.


