Installation
=================

InferPy is freely available at Pypi and it can be installed with the following command:

.. code:: bash

   $ python -m pip install inferpy


or equivalently

.. code:: bash

   $ pip install inferpy


The previous commands install our package only with the dependencies for a basic usage.
Instead, additional dependencies can be installed using the following keywords:

.. code:: bash

   $ pip install inferpy[gpu]               # running over GPUs

   $ pip install inferpy[visualization]     # including matplotlib

   $ pip install inferpy[datasets]          # for using datasets at inf.data


If we want to install InferPy including all the dependencies, use the keyword
``all``, that is:



.. code:: bash

   $ pip install inferpy[all]