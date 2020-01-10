
Bayesian Neural Networks
========================

Neural networks are powerful approximators. However, standard approaches
for learning this approximators does not take into account the inherent
uncertainty we may have when fitting a model.

.. code:: python3

    import logging, os
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import math
    import inferpy as inf
    import tensorflow_probability as tfp


Data
----

We use some fake data. As neural nets of even one hidden layer can be
universal function approximators, we can see if we can train a simple
neural network to fit a noisy sinusoidal data, like this:

.. code:: python3

    NSAMPLE = 100
    x_train = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_train = np.float32(np.random.normal(size=(NSAMPLE,1),scale=1.0))
    y_train = np.float32(np.sin(0.75*x_train)*7.0+x_train*0.5+r_train*1.0)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_train, y_train, marker='+', label='Training data')
    plt.legend();



.. image:: ../_static/img/notebooks/output_3_0.png


Training a neural network
-------------------------

We employ a simple feedforward network with 20 hidden units to try to
fit the data.

.. code:: python3

    NHIDDEN = 20

    nnetwork = tf.keras.Sequential([
        tf.keras.layers.Dense(NHIDDEN, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1)
    ])

    lossfunc = lambda y_out, y: tf.nn.l2_loss(y_out-y)

    nnetwork.compile(tf.train.AdamOptimizer(0.01), lossfunc)
    nnetwork.fit(x=x_train, y=y_train, epochs=1000)


.. parsed-literal::

    Epoch 1/1000
    100/100 [==============================] - 0s 2ms/sample - loss: 396.0843
    Epoch 2/1000
    100/100 [==============================] - 0s 106us/sample - loss: 362.0025
    Epoch 3/1000
    100/100 [==============================] - 0s 69us/sample - loss: 349.1164
    [...]
    Epoch 998/1000
    100/100 [==============================] - 0s 54us/sample - loss: 17.3651
    Epoch 999/1000
    100/100 [==============================] - 0s 85us/sample - loss: 18.4519
    Epoch 1000/1000
    100/100 [==============================] - 0s 87us/sample - loss: 18.3769




.. parsed-literal::

    <tensorflow.python.keras.callbacks.History at 0x12e6eb3c8>



We see that the neural network can fit this sinusoidal data quite well,
as expected.

.. code:: python3

    sess = tf.keras.backend.get_session()
    x_test = np.float32(np.arange(-10.5,10.5,0.1))
    x_test = x_test.reshape(x_test.size,1)
    y_test = sess.run(nnetwork(x_test))

    plt.figure(figsize=(8, 8))
    plt.plot(x_test, y_test, 'r-', label='Predictive mean');
    plt.scatter(x_train, y_train, marker='+', label='Training data')
    plt.xticks(np.arange(-10., 10.5, 4))
    plt.title('Standard Neural Network')
    plt.legend();



.. image:: ../_static/img/notebooks/output_7_0.png


However this model is unable to capture the uncertainty in the model.
For example, when making predictions about a single point (e.g. around
x=2.0) we can see we do not account about the inherent noise there is in
this predictions. In next section, we will what happen when we introduce
a Bayesian approach using InferPy.

Bayesian Learning of Neural Networks
------------------------------------

`Bayesian
modeling <http://mlg.eng.cam.ac.uk/zoubin/papers/NatureReprint15.pdf>`__
offers a systematic framework for reasoning about model uncertainty.
Instead of just learning point estimates, we’re going to learn a
distribution over variables that are consistent with the observed data.

In Bayesian learning, the weights of the network are
``random variables``. The output of the network is another
``random variable``. And the random variable of the output is the one
that implicitlyl defines the ``loss function``. So, when making Bayesian
learning we do not define ``loss functions``, we do define
``random variables``. For more information you can check `this
talk <https://www.cs.ox.ac.uk/people/yarin.gal/website/PDFs/2017_OReilly_talk.pdf>`__
and this `paper <https://arxiv.org/abs/1908.03442>`__.

In `Inferpy <https://inferpy.readthedocs.io>`__, defining a Bayesian
neural network is quite straightforward. First we define our neural
network using ``inf.layers.Sequential`` and layers of class
``tfp.layers.DenseFlipout``. Second, the input ``x`` and output ``y``
are also define as random variables. More precisely, the output ``y`` is
defined as a Gaussian random variable. The mean of the Gaussian is the
output of the neural network.

.. code:: python3

    @inf.probmodel
    def model(NHIDDEN):

        with inf.datamodel():
            x = inf.Normal(loc = tf.zeros([1]), scale = 1.0, name="x")

            nnetwork = inf.layers.Sequential([
                tfp.layers.DenseFlipout(NHIDDEN, activation=tf.nn.tanh),
                tfp.layers.DenseFlipout(1)
            ])

            y = inf.Normal(loc = nnetwork(x) , scale= 1., name="y")

To perform Bayesian learning, we resort the scalable variational methods
available in InferPy, which require the definition of a ``q`` model. For
details,see the documentation about `Inference in
Inferpy <https://inferpy.readthedocs.io/projects/develop/en/develop/notes/guideinference.html>`__.
For a deeper theoretical description, read this
`paper <https://arxiv.org/abs/1908.03442>`__. In this case, the q
variables approximating the NN are defined in a transparent way. For
that reason we define an empty q model.

.. code:: python3

    @inf.probmodel
    def qmodel():
        pass


.. code:: python3

    NHIDDEN=20

    p = model(NHIDDEN)
    q = qmodel()

    VI = inf.inference.VI(q, optimizer = tf.train.AdamOptimizer(0.01), epochs=5000)

    p.fit({"x": x_train, "y": y_train}, VI)


.. parsed-literal::


     0 epochs	 3624.52294921875....................
     200 epochs	 2872.20947265625....................
     400 epochs	 2204.5673828125....................
     600 epochs	 2003.1492919921875....................
     800 epochs	 1982.3792724609375....................
     1000 epochs	 1976.4678955078125....................
     1200 epochs	 1975.5140380859375....................
     1400 epochs	 1974.84765625....................
     1600 epochs	 1974.39501953125....................
     1800 epochs	 1973.7484130859375....................
     2000 epochs	 1974.3310546875....................
     2200 epochs	 1972.4315185546875....................
     2400 epochs	 1972.6212158203125....................
     2600 epochs	 1973.4249267578125....................
     2800 epochs	 1973.1256103515625....................
     3000 epochs	 1972.3126220703125....................
     3200 epochs	 1971.6856689453125....................
     3400 epochs	 1972.302734375....................
     3600 epochs	 1971.942626953125....................
     3800 epochs	 1970.769287109375....................
     4000 epochs	 1970.2620849609375....................
     4200 epochs	 1971.232666015625....................
     4400 epochs	 1969.9805908203125....................
     4600 epochs	 1970.2784423828125....................
     4800 epochs	 1969.9132080078125....................

As can be seen in the nex figure, the output of our model is not
deterministic. So, we can capture the uncertainty in the data. See for
example what happens now with the predictions at the point ``x=2.0``.
See also what happens with the uncertainty in out-of-range predictions.

.. code:: python3

    x_test = np.linspace(-20.5, 20.5, NSAMPLE).reshape(-1, 1)

    plt.figure(figsize=(8, 8))

    y_pred_list = []
    for i in range(100):
        y_test = p.posterior_predictive(["y"], data = {"x": x_test}).sample()
        y_pred_list.append(y_test)

    y_preds = np.concatenate(y_pred_list, axis=1)

    y_mean = np.mean(y_preds, axis=1)
    y_sigma = np.std(y_preds, axis=1)

    plt.plot(x_test, y_mean, 'r-', label='Predictive mean');
    plt.scatter(x_train, y_train, marker='+', label='Training data')
    plt.fill_between(x_test.ravel(),
                     y_mean + 2 * y_sigma,
                     y_mean - 2 * y_sigma,
                     alpha=0.5, label='Epistemic uncertainty')
    plt.xticks(np.arange(-20., 20.5, 4))
    plt.title('Bayesian Neural Network')
    plt.legend();



.. image:: ../_static/img/notebooks/output_16_0.png

