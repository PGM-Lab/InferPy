Requirements
================


Python
-------------



Currently, InferPy requires Python 2.7 or 3.x. For checking your default Python version, type:


.. code:: bash

    $ python --version

Travis tests are performed on versions 2.7, 3.5 and 3.6. Go to `https://www.python.org/ <https://www.python.org/>`_
for specific instructions for installing the Python interpreter in your system.


Edward
-------------

InferPy requires exactly the version 1.3.5 of `Edward <http://edwardlib.org>`_. You may check the installed
package version as follows.


.. code:: bash

    $ pip freeze | grep edward

Tensorflow
-----------------

`Tensorflow <http://www.tensorflow.org/>`_: from version 1.5 up to 1.7 (both included). For checking the installed tensorflow version, type:

.. code:: bash

    $ pip freeze | grep tensorflow

Numpy
----------------

`Numpy <http://www.numpy.org/>`_ 1.14 or higher is required. For checking the version of this package, type:


.. code:: bash

    $ pip freeze | grep numpy

