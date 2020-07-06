Comparison: Variational auto-encoder
===========================================================

Here we make a comparison between tensorflow-probability/Edward 2, Pyro and InferPy. As a running example, we will consider
a variational auto-encoder (VAE) trained with the MNIST dataset containing handwritten digits. For the inference, SVI method
will be used.


Setting up
-------------

First, we import the required packages and set the global variables. This code is common for the 3 different frameworks:

.. literalinclude:: ../../examples/edward/ed_vae_mnist.py
   :language: python3
   :lines: 1-28

Then, we can load and plot the MNIST dataset using the functionality provided by ``inferpy.data.mnist``.

.. literalinclude:: ../../examples/probzoo/vae_mnist.py
   :language: python3
   :lines: 30-35

The generated plot is shown in the figure below.

.. figure:: ../_static/img/mnist_train.png
   :alt: MNIST training data.
   :scale: 40 %
   :align: center



Model definition
--------------------

P and Q models are defined as functions creating random variables. In the case of the VAE model, we must also
define the neural networks for encoding and decoding. For simplicity, they are also defined as functions. The model
definitions using InferPy, Edward and Pyro are shown below.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/vae_mnist.py
         :language: python3
         :lines: 41-66

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_vae_mnist.py
         :language: python3
         :lines: 46-68

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_vae.py
         :language: python3
         :lines: 43-117


With InferPy we do not need to specify which is the size of the data (i.e., plateau or
datamodel construct). Instead, this will be automatically obtained at inference time.


With InferPy and Edward 2, models are defined as functions, though InferPy requires to use the decorator ``@inf.probmodel``. On the
other hand, even though  neural networks can be the same, in the Edward 2's code these are defined with a name as this
will be later used for access to the learned weights. The code in Pyro (adapted from the one in the `official documentation <https://pyro.ai/examples/vae.html>`_)
is quite different as a class structure is used.

Inference
---------------

Setting up the inference and batched data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Edward 2, before optimizing the variational parameters, we must: split the data into batches; create the instances of the P and Q
models; and finally build tensor for computing the variational **ELBO**, which represents the function that will be optimized.
The equivalent code using InferPy is much more simple, because most of the functionality is done transparently to the user:
we simply instantiate the P and Q models and the corresponding inference algorithm. Pyro's code also remains quite simple because
most of the inference details are also encapsulated. Yet the user is required to split the data into batches using the 
functionality in ``torch.utils.data.DataLoader``.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/vae_mnist.py
         :language: python3
         :lines: 78-82

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_vae_mnist.py
         :language: python3
         :lines: 89-104

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_vae.py
         :language: python3
         :lines: 127-138



Optimization loop
^^^^^^^^^^^^^^^^^^^^^^^

In variational inference, parameters are iteratively optimized. When using Edward 2, we must first specify TensorFlow optimizers and training objects. Then the loop is explicitly coded as shown below. With Pyro, the optimization loop must coded by calling ``svi.step()`` at each iteration. By contrast, with InferPy, we simply invoke the method ``probmodel.fit()`` which takes as input parameters the data and the inference algorithm object previously defined.

.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/vae_mnist.py
         :language: python3
         :lines: 88-89

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_vae_mnist.py
         :language: python3
         :lines: 112-127

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_vae.py
         :language: python3
         :lines: 147-165



Usage of the inferred model
----------------------------------

Once optimization is finished, we can use the model with the inferred parameters. For example, we
might obtain the hidden representation of the original data, which is done by passing such data through the decoder.
Edward does not provide any functionality for this purpose, so we will use TensorFlow code. With InferPy, this is done by simply using the method ``probmodel.posterior()`` as follows.
For this, Pyro requires to invoke the class method ``vae.encoder.forward`` which performs the forward propagation in the encoder NN.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/vae_mnist.py
         :language: python3
         :lines: 95-98

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_vae_mnist.py
         :language: python3
         :lines: 137-149

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_vae.py
         :language: python3
         :lines: 166-169




The result of plotting the hidden representation is:


.. figure:: ../_static/img/postz_cloud.png
   :alt: Plot of the hidden encoding.
   :scale: 60 %
   :align: center



We might be also interested in generating new digits, which implies passing some data in the hidden space
through the decoder. With InferPy we must just invoke the method ``probmodel.posterior_predictive()``.
In Pyro this is done by invoking the class method ``vae.encoder.forward`` which performs the forward propagation in the decoder NN.


.. tabs::

   .. group-tab:: InferPy

      .. literalinclude:: ../../examples/probzoo/vae_mnist.py
         :language: python3
         :lines: 120-121

   .. group-tab:: TFP/Edward 2

      .. literalinclude:: ../../examples/edward/ed_vae_mnist.py
         :language: python3
         :lines: 177-187

   .. group-tab:: Pyro

      .. literalinclude:: ../../examples/pyro/pyro_vae.py
         :language: python3
         :lines: 193-195



Some of the resulting images are shown below.

.. figure:: ../_static/img/mnist_gen.png
   :alt: MNIST generated data.
   :scale: 40 %
   :align: center


