Variational auto-encoder (VAE) in Edward and Inferpy
===========================================================


Setting up
-------------


.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 1-26

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 30-35




.. figure:: ../_static/img/mnist_train.png
   :alt: MNIST training data
   :scale: 40 %
   :align: center



Model definition
--------------------

.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 39-67

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 39-69


Inference
---------------

Setting up the inference and batched data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 75-104

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 76-82


Optimization loop
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 110-127

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 85-88



Usage of the inferred model
----------------------------------

.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 135-149

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 93-96



.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 175-188

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 117-121
