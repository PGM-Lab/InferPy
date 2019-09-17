
Bayesian Neural Networks
========================

Neural networks are powerful approximators. However, standard approaches
for learning this approximators does not take into account the inherent
uncertainty we may have when fitting a model.

.. code:: python3

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import math
    import inferpy as inf
    import warnings
    warnings.filterwarnings("ignore")

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

    x = tf.placeholder(dtype=tf.float32, shape=[None,1])
    y = tf.placeholder(dtype=tf.float32, shape=[None,1])
    
    
    NHIDDEN = 20
    W = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
    b = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
    
    W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
    b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
    
    hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
    y_out = tf.matmul(hidden_layer,W_out) + b_out
    
    lossfunc = tf.nn.l2_loss(y_out-y);
    
    
    train_op = tf.train.AdamOptimizer(0.01).minimize(lossfunc)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    NEPOCH = 1000
    for i in range(NEPOCH):
      sess.run(train_op,feed_dict={x: x_train, y: y_train})
      if i%100==0: 
            print(sess.run(lossfunc,feed_dict={x: x_train, y: y_train}))



.. parsed-literal::

    2310.9165
    958.04425
    747.9302
    504.03128
    279.17215
    137.36455
    75.68969
    55.536686
    49.636005
    47.68384


We see that the neural network can fit this sinusoidal data quite well,
as expected.

.. code:: python3

    x_test = np.float32(np.arange(-10.5,10.5,0.1))
    x_test = x_test.reshape(x_test.size,1)
    y_test = sess.run(y_out,feed_dict={x: x_test})
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_test, y_test, 'r-', label='Predictive mean');
    plt.scatter(x_train, y_train, marker='+', label='Training data')
    plt.xticks(np.arange(-10., 10.5, 4))
    plt.title('Standard Neural Network')
    plt.legend();
    sess.close()



.. image:: ../_static/img/notebooks/output_7_0.png


However this model is unable to capture the uncertainty in the model.
For example, when making predictions about a single point (e.g. around
x=2.0) we can see we do not account aobut the inherent noise there is in
this predictions. In next section, we will what happen when we introduce
a Bayesian approach using Inferpy.

Bayesian Learning of Neural Networks
------------------------------------

`Bayesian
modeling <http://mlg.eng.cam.ac.uk/zoubin/papers/NatureReprint15.pdf>`__
offers a systematic framework for reasoning about model uncertainty.
Instead of just learning point estimates, we’re going to learn a
distribution over variables that are consistent with the observed data.

In Bayesian learning, the weights of the network are
``random variables``. The output of the nework is another
``random variable``. And the random variable of the output is the one
that implicitlyl defines the ``loss function``. So, when making Bayesian
learning we do not define ``loss functions``, we do define
``random variables``. For more information you can check `this
talk <https://www.cs.ox.ac.uk/people/yarin.gal/website/PDFs/2017_OReilly_talk.pdf>`__
and this `paper <https://arxiv.org/abs/1908.03442>`__.

In `Inferpy <https://inferpy.readthedocs.io>`__, defining a Bayesian
neural network is quite straightforward. First we define our model,
where the weights of the neural network are defined as random variables.
Second, the input ``x`` and output ``y`` are also define as random
variables. More precisely, the output ``y`` is defined as a Gaussian
random varible. The mean of the Gaussian is the output of the neural
network.

.. code:: python3

    @inf.probmodel
    def model(NHIDDEN):
        W = inf.Normal(loc = tf.zeros([1,NHIDDEN]), scale=1., name="W")
        b = inf.Normal(loc = tf.zeros([1,NHIDDEN]), scale=1., name="b")
    
        W_out = inf.Normal(loc = tf.zeros([NHIDDEN,1]), scale=1., name="W_out")
        b_out = inf.Normal(loc = tf.zeros([1,1]), scale=1., name="b_out")
    
        with inf.datamodel():
            x = inf.Normal(loc = tf.zeros([1]), scale = 1.0, name="x")
            hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
            out = tf.matmul(hidden_layer,W_out) + b_out
            y = inf.Normal(loc = out , scale= 1., name="y")

To perform Bayesian learning, we resort the scalable variational methods
available in Inferpy, which require the definition of a ``q`` model. For
details,see the documentation about `Inference in
Inferpy <https://inferpy.readthedocs.io/projects/develop/en/develop/notes/guideinference.html>`__.
For a deeper theoretical despcription, read this
`paper <https://arxiv.org/abs/1908.03442>`__.

.. code:: python3

    @inf.probmodel
    def qmodel(NHIDDEN):
        W_loc = inf.Parameter(tf.random_normal([1,NHIDDEN], 0.0, 0.05, dtype=tf.float32))
        b_loc = inf.Parameter(tf.random_normal([1,NHIDDEN], 0.0, 0.05, dtype=tf.float32))
        W_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([1,NHIDDEN], -10., stddev=0.05 ,dtype=tf.float32)))+0.01
        b_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([1,NHIDDEN],  -10., stddev=0.05 ,dtype=tf.float32)))
    
        qW = inf.Normal(W_loc, scale = W_scale, name="W")
        qb = inf.Normal(b_loc, scale = b_scale, name="b")
    
        W_out_loc = inf.Parameter(tf.random_normal([NHIDDEN,1], 0.0, 0.05, dtype=tf.float32))
        b_out_loc = inf.Parameter(tf.random_normal([1,1], 0.0, 0.05, dtype=tf.float32))
        W_out_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([NHIDDEN,1],  -10., stddev=0.05, dtype=tf.float32)))
        b_out_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([1,1],  -10., stddev=0.05, dtype=tf.float32)))
    
        qW_out = inf.Normal(W_out_loc, scale = W_out_scale, name="W_out")
        qb_out = inf.Normal(b_out_loc, scale = b_out_scale, name="b_out")

.. code:: python3

    NHIDDEN=20
    
    p = model(NHIDDEN)
    q = qmodel(NHIDDEN)
    
    VI = inf.inference.VI(q, optimizer = tf.train.AdamOptimizer(0.01), epochs=5000)
    
    p.fit({"x": x_train, "y": y_train}, VI)


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0917 11:54:32.418200 4669543872 deprecation_wrapper.py:119] From /Users/rcabanas/venv/InferPy/lib/python3.6/site-packages/inferpy/models/prob_model.py:63: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    W0917 11:54:32.500988 4669543872 deprecation_wrapper.py:119] From /Users/rcabanas/venv/InferPy/lib/python3.6/site-packages/inferpy/models/random_variable.py:420: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.
    
    W0917 11:54:32.532026 4669543872 deprecation_wrapper.py:119] From /Users/rcabanas/venv/InferPy/lib/python3.6/site-packages/inferpy/util/tf_graph.py:63: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0917 11:54:32.768965 4669543872 deprecation_wrapper.py:119] From /Users/rcabanas/venv/InferPy/lib/python3.6/site-packages/inferpy/models/prob_model.py:136: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.
    
    W0917 11:54:33.819847 4669543872 deprecation.py:323] From /Users/rcabanas/venv/InferPy/lib/python3.6/site-packages/inferpy/util/interceptor.py:21: Variable.load (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.
    Instructions for updating:
    Prefer Variable.assign which has equivalent behavior in 2.X.
    W0917 11:54:35.361786 4669543872 deprecation.py:323] From /Users/rcabanas/venv/InferPy/lib/python3.6/site-packages/tensorflow/python/ops/array_ops.py:1354: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where


.. parsed-literal::

    
     0 epochs	 4921.72021484375....................
     200 epochs	 3732.502197265625....................
     400 epochs	 3457.382568359375....................
     600 epochs	 3007.405517578125....................
     800 epochs	 2890.67822265625....................
     1000 epochs	 2874.74169921875....................
     1200 epochs	 2843.31591796875....................
     1400 epochs	 2855.461669921875....................
     1600 epochs	 2830.63818359375....................
     1800 epochs	 2847.599853515625....................
     2000 epochs	 2833.72802734375....................
     2200 epochs	 2856.710693359375....................
     2400 epochs	 2833.2412109375....................
     2600 epochs	 2834.54638671875....................
     2800 epochs	 2854.3798828125....................
     3000 epochs	 2742.443359375....................
     3200 epochs	 2590.494140625....................
     3400 epochs	 2575.948974609375....................
     3600 epochs	 2501.843994140625....................
     3800 epochs	 2500.095703125....................
     4000 epochs	 2468.91943359375....................
     4200 epochs	 2457.67626953125....................
     4400 epochs	 2523.811279296875....................
     4600 epochs	 2471.731689453125....................
     4800 epochs	 2469.966064453125....................

As can be seen in the nex figure, the output of our model is not
deterministic. So, we can caputure the uncertainty in the data. See for
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

