Getting Started:
================

Installation
-----------------

Install InferPy from PyPI:

.. code:: bash

   $ python -m pip install inferpy





30 seconds to InferPy
--------------------------

The core data structures of InferPy is a **probabilistic model**,
defined as a set of **random variables** with a conditional dependency
structure. A **random varible** is an object
parameterized by a set of tensors.

Let's look at a simple non-linear **probabilistic component analysis** model (NLPCA). Graphically the model can
be defined as follows, 

.. figure:: ../_static/img/nlpca.png
   :alt: Non-linear PCA
   :scale: 60 %
   :align: center
   
   Non-linear PCA

We start by importing the required packages and defining the constant parameters in the model.

.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 1-13

A model can be defined by decorating any function with ``@inf.probmodel``. The model is fully specified by
the variables defined inside this function:

.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 14-22


The construct ``with inf.datamodel()``, which resembles to the **plateau notation**, will replicate
N times the variables enclosed, where N is the size of our data.


In the previous model, the input argument ``decoder`` must be a function implementing a neural network.
This might be defined outside the model as follows.


.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 27-30


Now, we can instantiate our model and obtain samples (from the prior distributions).

.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 48-53




In variational inference, we must defined a Q-model as follows.


.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 36-44



Afterwards, we define the parameters of our inference algorithm and fit the data to the model.



.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 62-66


The inference method can be further configure. But, as in Keras, a core
principle is to try make things reasonably simple, while allowing the
user the full control if needed.



Finally, we might extract the posterior of ``z``, which is basically the hidden representation
of our data.



.. literalinclude:: ../../examples/docs/getting30s/1.py
   :language: python3
   :lines: 71-73
