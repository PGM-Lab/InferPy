
Mixture Density Networks
========================

Mixture density networks (MDN) (Bishop, 1994) are a class of models
obtained by combining a conventional neural network with a mixture
density model.

.. code:: ipython3

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    
    import inferpy as inf
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import tensorflow as tf
    import tensorflow_probability as tfp
    
    from inferpy import Categorical, Mixture, Normal
    from scipy import stats
    from sklearn.model_selection import train_test_split


.. parsed-literal::

    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


.. code:: ipython3

    def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
      """Plots the mixture of Normal models to axis=ax comp=True plots all
      components of mixture model
      """
      x = np.linspace(-10.5, 10.5, 250)
      final = np.zeros_like(x)
      for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
        temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
        final = final + temp
        if comp:
          ax.plot(x, temp, label='Normal ' + str(i))
      ax.plot(x, final, label='Mixture of Normals ' + label)
      ax.legend(fontsize=13)
    
    
    def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
      """Draws samples from mixture model.
    
      Returns 2 d array with input X and sample from prediction of mixture model.
      """
      samples = np.zeros((amount, 2))
      n_mix = len(pred_weights[0])
      to_choose_from = np.arange(n_mix)
      for j, (weights, means, std_devs) in enumerate(
              zip(pred_weights, pred_means, pred_std)):
        index = np.random.choice(to_choose_from, p=weights)
        samples[j, 1] = np.random.normal(means[index], std_devs[index], size=1)
        samples[j, 0] = x[j]
        if j == amount - 1:
          break
      return samples

Data
----

We use the same toy data from `David Ha’s blog
post <http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/>`__,
where he explains MDNs. It is an inverse problem where for every input
:math:`x_n` there are multiple outputs :math:`y_n`.

.. code:: ipython3

    def build_toy_dataset(N):
      y_data = np.random.uniform(-10.5, 10.5, N).astype(np.float32)
      r_data = np.random.normal(size=N).astype(np.float32)  # random noise
      x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
      x_data = x_data.reshape((N, 1))
      return x_data, y_data
    
    import random 
    
    tf.random.set_random_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    #inf.setseed(42)
    
    N = 5000  # number of data points
    D = 1  # number of features
    K = 20  # number of mixture components
    
    X_train, y_train = build_toy_dataset(N)
    
    print("Size of features in training data: {}".format(X_train.shape))
    print("Size of output in training data: {}".format(y_train.shape))
    print("Size of features in test data: {}".format(X_test.shape))
    print("Size of output in test data: {}".format(y_test.shape))
    sns.regplot(X_train, y_train, fit_reg=False)
    plt.show()


.. parsed-literal::

    Size of features in training data: (5000, 1)
    Size of output in training data: (5000,)
    Size of features in test data: (5000, 1)
    Size of output in test data: (5000,)



.. image:: output_4_1.png


Fitting a Neural Network
------------------------

We could try to fit a neural network over this data set. However, in
this data set for each x value there are multiple y values. So, things
do not work as should be using standard neural networks.

Let’s define first the neural network. We use ``tf.layers`` to construct
neural networks. We specify a three-layer network with 15 hidden units
for each hidden layer.

.. code:: ipython3

    def neural_network(X):
      # 2 hidden layers with 15 hidden units
      net = tf.layers.dense(X, 15, activation=tf.nn.relu)
      net = tf.layers.dense(net, 15, activation=tf.nn.relu)
      out = tf.layers.dense(net, 1, activation=None)
      return out

Let’s now try to fit the neural network to the data

.. code:: ipython3

    x = tf.placeholder(dtype=tf.float32, shape=[None,1])
    y = tf.placeholder(dtype=tf.float32, shape=[None])
    
    y_out = neural_network(x)
    
    lossfunc = tf.nn.l2_loss(y_out-y);
    
    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(lossfunc)
    
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    NEPOCH = 100
    for i in range(NEPOCH):
      sess.run(train_op,feed_dict={x: X_train, y: y_train})
      if i%10==0:
            print(sess.run(lossfunc,feed_dict={x: X_train, y: y_train}))  
    
    y_test = sess.run(y_out,feed_dict={x: X_test})
    
    plt.figure(figsize=(8, 8))
    plt.plot(X_train,y_train,'ro',X_test,y_test,'bo',alpha=0.3)
    plt.show()
    
    sess.close()


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0821 06:22:38.783931 140736636462016 deprecation.py:323] From <ipython-input-4-3ee7d449962f>:4: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dense instead.
    W0821 06:22:38.798621 140736636462016 deprecation.py:506] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
    Instructions for updating:
    Call initializer instance with the dtype argument instead of passing it to the constructor
    W0821 06:22:39.369898 140736636462016 deprecation.py:323] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow/python/util/tf_should_use.py:193: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.


.. parsed-literal::

    504323700.0
    462389000.0
    462367100.0
    462353200.0
    462342900.0
    462338880.0
    462338700.0
    462338370.0
    462338180.0
    462338200.0



.. image:: output_9_2.png


As can be seen, the neural network is not able to fit this data set

Mixture Density Network (MDN)
-----------------------------

We use a MDN with a mixture of 20 normal distributions parameterized by
a feedforward network. That is, the membership probabilities and
per-component mean and standard deviation are given by the output of a
feedforward network.

We define our probabilistic model using ``Inferpy`` constructs.
Specifically, we use the ``MixtureSameFamily`` distribution, where the
the parameters of this network are provided by our feedforwrad network.

.. code:: ipython3

    def neural_network(X):
      """loc, scale, logits = NN(x; theta)"""
      # 2 hidden layers with 15 hidden units
      net = tf.layers.dense(X, 15, activation=tf.nn.relu)
      net = tf.layers.dense(net, 15, activation=tf.nn.relu)
      locs = tf.layers.dense(net, K, activation=None)
      scales = tf.layers.dense(net, K, activation=tf.exp)
      logits = tf.layers.dense(net, K, activation=None)
      return locs, scales, logits
    
    
    @inf.probmodel
    def mdn():
        with inf.datamodel():
            x = inf.Normal(loc = tf.ones([D]), scale = 1.0, name="x")
            locs, scales, logits = neural_network(x)
            y = inf.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=logits), components_distribution=tfp.distributions.Normal(loc=locs, scale=scales+0.01), name="y")
        
    m = mdn()


.. parsed-literal::

    W0821 06:25:57.125390 140736636462016 deprecation_wrapper.py:119] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/inferpy/models/prob_model.py:62: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    W0821 06:25:57.135653 140736636462016 deprecation_wrapper.py:119] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/inferpy/util/tf_graph.py:63: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0821 06:25:57.152807 140736636462016 deprecation_wrapper.py:119] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/inferpy/models/random_variable.py:430: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    W0821 06:25:57.342686 140736636462016 deprecation.py:323] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/tensorflow_probability/python/internal/distribution_util.py:493: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0821 06:25:57.385910 140736636462016 deprecation_wrapper.py:119] From /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/inferpy/models/prob_model.py:128: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    


Note that we use the ``MixtureSameFamily`` random variable. It collapses
out the membership assignments for each data point and makes the model
differentiable with respect to all its parameters. It takes a
``Categorical`` random variable as input—denoting the probability for
each cluster assignment—as well as ``components``, which is a list of
individual distributions to mix over.

For more background on MDNs, take a look at `Christopher Bonnett’s blog
post <http://cbonnett.github.io/MDN.html>`__ or at Bishop (1994).

Inference
---------

We train the MDN model. For details, see the documentation about
`Inference in
Inferpy <https://inferpy.readthedocs.io/projects/develop/en/develop/notes/guideinference.html>`__

.. code:: ipython3

    @inf.probmodel
    def qmodel():
        return;
    
    VI = inf.inference.VI(qmodel(), epochs=2000)
    m.fit({"y": y_train, "x":X_train}, VI)


.. parsed-literal::

    /Users/andresmasegosa/Dropbox/infer/tmp/inferpy/lib/python3.6/site-packages/inferpy/models/prob_model.py:179: UserWarning: Fit was called before. This will restart the inference method and                 re-build the expanded model.
      re-build the expanded model.")


.. parsed-literal::

    
     0 epochs	 133375.90625....................
     200 epochs	 113701.6796875....................
     400 epochs	 110918.515625....................
     600 epochs	 108761.9453125....................
     800 epochs	 106857.3828125....................
     1000 epochs	 106288.171875....................
     1200 epochs	 106097.1171875....................
     1400 epochs	 105861.578125....................
     1600 epochs	 105749.421875....................
     1800 epochs	 105694.640625....................

After training, we can now see how the same network embbeded in a
mixture model is able to perfectly capture the training data.

.. code:: ipython3

    X_test, y_test = build_toy_dataset(N)
    
    y_pred = m.posterior_predictive(["y"], data = {"x": X_test}).sample()
    
    
    plt.figure(figsize=(8, 8))
    sns.regplot(X_test, y_test, fit_reg=False)
    sns.regplot(X_test, y_pred, fit_reg=False)
    plt.show()



.. image:: output_17_0.png


Acknowledgments
---------------

This tutorial is inspired by `David Ha’s blog
post <http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/>`__
and `Edward’s
tutorial <http://edwardlib.org/tutorials/mixture-density-network>`__.
