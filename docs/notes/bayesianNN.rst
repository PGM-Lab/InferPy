
Bayesian Neural Networks
========================

Neural networks are powerful approximators. However, standard approaches
for learning this approximators does not take into account the inherent
uncertainty we may have when fitting a model.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import math
    import inferpy as inf


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


Data
----

We use some fake data. As neural nets of even one hidden layer can be
universal function approximators, we can see if we can train a simple
neural network to fit a noisy sinusoidal data, like this:

.. code:: ipython3

    NSAMPLE = 100
    x_train = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_train = np.float32(np.random.normal(size=(NSAMPLE,1),scale=1.0))
    y_train = np.float32(np.sin(0.75*x_train)*7.0+x_train*0.5+r_train*1.0)
    
    plt.figure(figsize=(8, 8))
    plot_out = plt.plot(x_train,y_train,'ro',markersize=5)
    plt.show()



.. image:: ../_static/img/notebooks/output_3_0.png


Training a neural network
-------------------------

We employ a simple feedforward network with 20 hidden units to try to
fit the data.

.. code:: ipython3

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

    1329.1733
    724.98596
    504.65738
    290.93298
    166.82144
    108.051575
    79.41223
    65.34183
    58.50912
    55.22774


We see that the neural network can fit this sinusoidal data quite well,
as expected.

.. code:: ipython3

    x_test = np.float32(np.arange(-10.5,10.5,0.1))
    x_test = x_test.reshape(x_test.size,1)
    y_test = sess.run(y_out,feed_dict={x: x_test})
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_train,y_train,'ro')
    plt.plot(x_test,y_test,'bo',markersize=1)
    plt.xticks(np.arange(-10., 10, 2))
    plt.show()
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

In Inferpy, defining a Bayesian neural network is quite straightforward.
First we define our model, where the weights of the neural network are
defined as random variables. Second, the input ``x`` and output ``y``
are also define as random variables. More precisely, the output ``y`` is
defined as a Gaussian random varible. The mean of the Gaussian is the
output of the neural network, and the scale (or standard deviation) of
the Gaussian is also learnt from data. In this case, we do not follow a
Bayesian treatment of the parameter, and we perform a simpler maximum
likelihood estimate.

.. code:: ipython3

    @inf.probmodel
    def model1(NHIDDEN):
        W = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
        b = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
    
        W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
        b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
    
        scale = tf.nn.softplus(tf.Variable(tf.random_normal([1], -5., stddev=0.05, dtype=tf.float32)))
    
        with inf.datamodel():
            x = inf.Normal(loc = tf.ones([1]), scale = 1.0, name="x")
            hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
            out = tf.matmul(hidden_layer,W_out) + b_out
            y = inf.Normal(loc = out , scale= scale, name="y")
    
    
    @inf.probmodel
    def model2(NHIDDEN):
        W = inf.Normal(loc = tf.zeros([1,NHIDDEN]), scale=1., name="W")
        b = inf.Normal(loc = tf.zeros([1,NHIDDEN]), scale=1., name="b")
    
        W_out = inf.Normal(loc = tf.zeros([NHIDDEN,1]), scale=1., name="W_out")
        b_out = inf.Normal(loc = tf.zeros([1,1]), scale=1., name="b_out")
    
        scale = tf.nn.softplus(inf.Parameter(-5., name="scale"))
    
    
        with inf.datamodel():
            x = inf.Normal(loc = tf.zeros([1]), scale = 1.0, name="x")
            hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
            out = tf.matmul(hidden_layer,W_out) + b_out
            y = inf.Normal(loc = out , scale= scale, name="y")
    
    @inf.probmodel
    def model3(NHIDDEN):
        W = inf.Normal(loc = tf.zeros([1,1]), scale=1.0, name="W")
        b = inf.Normal(loc = tf.zeros([1,1]), scale=1.0, name="b")
        
        with inf.datamodel():
            x = inf.Normal(loc = tf.zeros([1]), scale = 1.0, name="x")
            out = tf.matmul(x, W) + b
            y = inf.Normal(loc = out , scale= 0.01, name="y")


To perform Bayesian learning, we resort the scalable variational methods
available in Inferpy, which require the definition of a ``q`` model. For
details,see the documentation about `Inference in
Inferpy <https://inferpy.readthedocs.io/projects/develop/en/develop/notes/guideinference.html>`__.
For a deeper theoretical despcription, read this
`paper <https://arxiv.org/abs/1908.03442>`__.

.. code:: ipython3

    @inf.probmodel
    def qmodel1(NHIDDEN):
        return;
    
    @inf.probmodel
    def qmodel2(NHIDDEN):
        W_loc = inf.Parameter(tf.random_normal([1,NHIDDEN], 0.0, 0.05, dtype=tf.float32))
        b_loc = inf.Parameter(tf.random_normal([1,NHIDDEN], 0.0, 0.05, dtype=tf.float32))
        W_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([1,NHIDDEN], -5., stddev=0.05 ,dtype=tf.float32)))+0.01
        b_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([1,NHIDDEN],  -5., stddev=0.05 ,dtype=tf.float32)))
    
        qW = inf.Normal(W_loc, scale = W_scale, name="W")
        qb = inf.Normal(b_loc, scale = b_scale, name="b")
    
        W_out_loc = inf.Parameter(tf.random_normal([NHIDDEN,1], 0.0, 0.05, dtype=tf.float32))
        b_out_loc = inf.Parameter(tf.random_normal([1,1], 0.0, 0.05, dtype=tf.float32))
        W_out_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([NHIDDEN,1],  -5., stddev=0.05, dtype=tf.float32)))
        b_out_scale = tf.nn.softplus(inf.Parameter(tf.random_normal([1,1],  -10., stddev=0.05, dtype=tf.float32)))
    
        qW_out = inf.Normal(W_out_loc, scale = W_out_scale, name="W_out")
        qb_out = inf.Normal(b_out_loc, scale = b_out_scale, name="b_out")
        
    @inf.probmodel
    def qmodel3(NHIDDEN):
        W_loc = inf.Parameter(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
        b_loc = inf.Parameter(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
        W_scale = tf.nn.softplus(inf.Parameter(tf.random_uniform([1,1], dtype=tf.float32)))+0.01
        b_scale = tf.nn.softplus(inf.Parameter(tf.random_uniform([1,1], dtype=tf.float32)))+0.01
    
        qW = inf.Normal(W_loc, scale = W_scale, name="W")
        qb = inf.Normal(b_loc, scale = b_scale, name="b")
        


.. code:: ipython3

    NHIDDEN=20
    
    p = model2(NHIDDEN)
    q = qmodel2(NHIDDEN)
    
    VI = inf.inference.VI(q, optimizer = tf.train.AdamOptimizer(0.01), epochs=10000)
    
    p.fit({"x": x_train, "y": y_train}, VI)


.. parsed-literal::

    
     0 epochs	 35735312.0....................
     200 epochs	 2845490.5....................
     400 epochs	 1061176.5....................
     600 epochs	 600484.5625....................
     800 epochs	 375528.15625....................
     1000 epochs	 242850.59375....................
     1200 epochs	 163767.578125....................
     1400 epochs	 115047.953125....................
     1600 epochs	 83717.1796875....................
     1800 epochs	 62864.21875....................
     2000 epochs	 49013.87890625....................
     2200 epochs	 38575.46875....................
     2400 epochs	 31607.583984375....................
     2600 epochs	 27029.865234375....................
     2800 epochs	 22581.046875....................
     3000 epochs	 19761.11328125....................
     3200 epochs	 17654.70703125....................
     3400 epochs	 15910.0341796875....................
     3600 epochs	 14671.56640625....................
     3800 epochs	 13384.9501953125....................
     4000 epochs	 12463.4658203125....................
     4200 epochs	 11950.05859375....................
     4400 epochs	 10982.8408203125....................
     4600 epochs	 10300.392578125....................
     4800 epochs	 9672.2470703125....................
     5000 epochs	 9145.77734375....................
     5200 epochs	 8932.708984375....................
     5400 epochs	 8191.85302734375....................
     5600 epochs	 8209.7236328125....................
     5800 epochs	 7552.75732421875....................
     6000 epochs	 6912.53955078125....................
     6200 epochs	 6559.40576171875....................
     6400 epochs	 6242.12353515625....................
     6600 epochs	 5923.5263671875....................
     6800 epochs	 5734.498046875....................
     7000 epochs	 5390.36083984375....................
     7200 epochs	 5145.8193359375....................
     7400 epochs	 4881.78173828125....................
     7600 epochs	 4561.38916015625....................
     7800 epochs	 4631.59033203125....................
     8000 epochs	 4278.81884765625....................
     8200 epochs	 4026.1162109375....................
     8400 epochs	 3859.720703125....................
     8600 epochs	 3713.70263671875....................
     8800 epochs	 3599.844970703125....................
     9000 epochs	 3435.13916015625....................
     9200 epochs	 3320.072265625....................
     9400 epochs	 3246.248046875....................
     9600 epochs	 3085.03125....................
     9800 epochs	 3025.88232421875....................

As can be seen in the nex figure, the output of our model is not
deterministic. So, we can caputure the uncertainty in the data. See for
example what happens now with the predictions at the point ``x=2.0``.

.. code:: ipython3

    plt.figure(figsize=(8, 8))
    for i in range(1000):
        x_test = np.float32(np.random.uniform(-12.5, 12.5, (1, NSAMPLE))).T
        y_test = p.posterior_predictive(["y"], data = {"x": x_test}).sample()
        plt.plot(x_test,y_test,'bo',markersize=1)
    plt.plot(x_train,y_train,'ro',markersize=5)
    plt.xticks(np.arange(-10., 10, 2))
    plt.show()



.. image:: ../_static/img/notebooks/output_16_0.png

