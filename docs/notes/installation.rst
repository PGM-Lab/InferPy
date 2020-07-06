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


If we want to install InferPy including all the dependencies (for CPU only), use the keyword
``all``, that is:


.. code:: bash

   $ pip install inferpy[all]


Similarly, for installing all the dependencies including those for running over GPUs, use the keyword ``all-gpu``:


.. code:: bash

   $ pip install inferpy[all-gpu]



A video tutorial about the installation can be found `here <notes/videotutorials.html>`_.
