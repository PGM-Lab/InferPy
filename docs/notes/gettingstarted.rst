Getting Started
=====================================================


30 seconds to InferPy
--------------------------------------

The core data structures of InferPy is a a **probabilistic model**,
defined as a set of **random variables** with a conditional independence
structure. Like in Edward, a **random varible** is an object
parameterized by a set of tensors.

Let's look at a simple examle. We start defining the **prior** of the
parameters of a **mixture of Gaussians** model:

.. code:: python

    import numpy as np
    import inferpy as inf
    from inferpy.models import Normal, InverseGamma, Dirichlet

    # K defines the number of components. 
    K=10
    with inf.replicate(size = K)
        #Prior for the means of the Gaussians 
        mu = Normal(loc = 0, scale = 1)
        #Prior for the precision of the Gaussians 
        sigma = InverseGamma(concentration = 1, rate = 1)
        
    #Prior for the mixing proportions
    p = Dirichlet(np.ones(K))

InferPy supports the definition of **plateau notation** by using the
construct :python:`with inf.replicate(size = K)`, which replicates K times the
random variables enclosed within this anotator. Every replicated
variable is assumed to be **independent**.

This ``with inf.replicate(size = N)`` construct is also usefuel when
defining the model for the data:

.. code:: python

    # Number of observations
    N = 1000
    #data Model
    with inf.replicate(size = N)
        # Sample the component indicator of the mixture. This is a latent variable that can not be observed
        z_n = Multinomial(probs = p)
        # Sample the observed value from the Gaussian of the selected component.  
        x_n = Normal(loc = inf.gather(mu,z_n), scale = inf.gather(sigma,z_n), observed = true)

As commented above, the variable ``z_n`` and ``x_n`` are surrounded by a
``with`` statement to inidicate that the defined random variables will
be reapeatedly used in each data sample. In this case, every replicated
variable is conditionally idependent given the variables mu and sigma
defined outside the ``with`` statement.

Once the random variables of the model are defined, the probablitic
model itself can be created and compiled. The probabilistic model
defines a joint probability distribuiton over all these random
variables.

.. code:: python

    from inferpy import ProbModel
    probmodel = ProbModel(vars = [p,mu,sigma,z_n,x_n]) 
    probmodel.compile(infMethod = 'KLqp')

During the model compilation we specify different inference methods that
will be used to learn the model.

.. code:: python

    from inferpy import ProbModel
    probmodel = ProbModel(vars = [p,mu,sigma,z_n,x_n]) 
    probmodel.compile(infMethod = 'MCMC')

The inference method can be further configure. But, as in Keras, a core
principle is to try make things reasonbly simple, while allowing the
user the full control if needed.

.. code:: python

    from keras.optimizers import SGD
    probmodel = ProbModel(vars = [p,mu,sigma,z_n,x_n]) 
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    infklqp = inf.inference.KLqp(optimizer = sgd, loss="ELBO")
    probmodel.compile(infMethod = infklqp)

Every random variable object is equipped with methods such as
``log_prob()`` and ``sample()``. Similarly, a probabilistic model is also
equipped with the same methods. Then, we can sample data from the model
anbd compute the log-likelihood of a data set:

.. code:: python

    data = probmodel.sample(size = 100)
    log_like = probmodel.log_prob(data)

Of course, you can fit your model with a given data set:

.. code:: python

    probmodel.fit(data_training, epochs=10)

Update your probablistic model with new data using the Bayes' rule:

.. code:: python

    probmodel.update(new_data)

Query the posterior over a given random varible:

.. code:: python

    mu_post = probmodel.posterior(mu)

Evaluate your model according to a given metric:

.. code:: python

    log_like = probmodel.evaluate(test_data, metrics = ['log_likelihood'])

Or compute predicitons on new data

.. code:: python

    cluster_assignments = probmodel.predict(test_data, targetvar = z_n)

--------------

Guiding Principles
------------------

-  InferPy's probability distribuions are mainly inherited from
   TensorFlow Distribuitons package. InferPy's API is fully compatible
   with tf.distributions' API. The 'shape' argument was added as a
   simplifing option when defining multidimensional distributions.
-  InferPy directly relies on top of Edward's inference engine and
   includes all the inference algorithms included in this package. As
   Edward's inference engine relies on TensorFlow computing engine,
   InferPy also relies on it too.
-  InferPy seamsly process data contained in a numpy array, Tensorflow's
   tensor, Tensorflow's Dataset (tf.Data API) or Apache Spark's
   DataFrame.
-  InferPy also includes novel distributed statistical inference
   algorithms by combining Tensorflow and Apache Spark computing
   engines.

--------------



Guide to Building Probabilistic Models
-----------------------------------------------------------------

InferPy focuses on *hirearchical probabilistic models* which usually are
structured in two different layers:

-  A **prior model** defining a joint distribution :math:`p(\theta)`
   over the global parameters of the model, :math:`\theta`.
-  A **data or observation model** defining a joint conditional
   distribution :math:`p(x,h|\theta)` over the observed quantities
   :math:`x` and the the local hidden variables :math:`h` governing the
   observation :math:`x`. This data model should be specified in a
   single-sample basis. There are many models of interest without local
   hidden variables, in that case we simply specify the conditional
   :math:`p(x|\theta)`. More flexible ways of defining the data model
   can be found in ?.

This is how a mixture of Gaussians models is denfined in InferPy:

.. code:: python

    import numpy as np
    import inferpy as inf
    from inferpy.models import Normal, InverseGamma, Dirichlet

    # K defines the number of components. 
    K=10
    #Prior for the means of the Gaussians 
    mu = Normal(loc = 0, scale = 1, shape=[K,d])
    #Prior for the precision of the Gaussians 
    invgamma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
    #Prior for the mixing proportions
    theta = Dirichlet(np.ones(K))

    # Number of observations
    N = 1000
    #data Model
    with inf.replicate(size = N)
        # Sample the component indicator of the mixture. This is a latent variable that can not be observed
        z_n = Multinomial(probs = theta)
        # Sample the observed value from the Gaussian of the selected component.  
        x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(invgamma,z_n), observed = true)

    #Probabilistic Model
    probmodel = ProbModel(prior = [p,mu,sigma,z_n,x_n]) 
    probmodel.compile()

The ``with inf.replicate(size = N)`` sintaxis is used to replicate the
random variables contained within this construct. It follows from the
so-called *plateau notation* to define the data generation part of a
probabilistic model. Every replicated variable is **conditionally
idependent** given the previous random variables (if any) defined
outside the **with** statement.

Internally, ``with inf.replicate(size = N)`` construct modifies the
random variable shape by adding an extra dimension. For the above
example, z\_n's shape is [N,1], and x\_n's shape is [N,d].

Following Edward's approach, a random variable :math:`x` is an object
parametrized by a tensor :math:`\theta` (i.e. a TensorFlow's tensor or
numpy's ndarray). The number of random variables in one object is
determined by the dimensions of its parameters (like in Edward) or by
the 'shape' or 'dim' argument (inspired by PyMC3 and Keras):

.. code:: python

    # vector of 5 univariate standard normals
    x  = Normal(loc = 0, scale = 1, dim = 5) 

    # vector of 5 univariate standard normals
    x  = Normal(loc = np.zeros(5), scale = np.ones(5)) 

    # vector of 5 univariate standard normals
    x = Normal (loc = 0, scale = 1, shape = [5,1])

The ``with inf.replicate(size = N)`` sintaxis can also be used to define
multi-dimensional objects, the following code is also equivalent to the
above ones:

.. code:: python

    # vector of 5 univariate standard normals
    with inf.replicate(size = 5)
        x = Normal (loc = 0, scale = 1)

More detailed inforamtion about the semantics of
``with inf.replicate(size = N)`` can be found in ?. Examples of using
this construct to define more expressive and complex models can be found
in ?.

Multivariate distributions can be defined similarly. Following Edward's
approach, the multivariate dimension is the innermost (right-most)
dimension of the parameters.

.. code:: python

    # 2 x 3 matrix of K-dimensional multivariate normals
    x  = MultivariateNormal(loc = np.zeros([2,3,K]), scale = np.ones([2,3,K,K]), observed = true) 

    # 2 x 3 matrix of K-dimensional multivariate normals
    y = MultivariateNormal (loc = np.zeros(K), scale = np.ones([K,K]), shape = [2,3], observed = true)

The argument **observed = true** in the constructor of a random variable
is used to indicate whether a variable is observable or not.

A **probabilistic model** defines a joint distribution over observable
and non-observable variables, :math:`p(theta,mu,sigma,z_n, x_n)` for the
running example,

.. code:: python

    from inferpy import ProbModel
    probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
    probmodel.compile()

The model must be **compiled** before it can be used.

Like any random variable object, a probabilistic model is equipped with
methods such as *log\_prob()* and *sample()*. Then, we can sample data
from the model anbd compute the log-likelihood of a data set:

.. code:: python

    data = probmodel.sample(size = 1000)
    log_like = probmodel.log_prob(data)

Folowing Edward's approach, a random variable :math:`x` is associated to
a tensor :math:`x^*` in the computational graph handled by TensorFlow,
where the computations takes place. This tensor :math:`x^*` contains the
samples of the random variable :math:`x`, i.e.
:math:`x^*\sim p(x|\theta)`. In this way, random variables can be
involved in expressive deterministic operations. For example, the
following piece of code corresponds to a zero inflated linear regression
model

.. code:: python


    #Prior
    w = Normal(0, 1, dim=d)
    w0 = Normal(0, 1)
    p = Beta(1,1)

    #Likelihood model
    with inf.replicate(size = 1000):
        x = Normal(0,1000, dim=d, observed = true)
        h = Binomial(p)
        y0 = Normal(w0 + inf.matmul(x,w, transpose_b = true), 1),
        y1 = Delta(0.0)
        y = Deterministic(h*y0 + (1-h)*y1, observed = true)

    probmodel = ProbModel(vars = [w,w0,p,x,h,y0,y1,y]) 
    probmodel.compile()
    data = probmodel.sample(size = 1000)
    probmodel.fit(data)

A special case, it is the inclusion of deep neural networks within our
probabilistic model to capture complex non-linear dependencies between
the random variables. This is extensively treated in the the Guide to
Bayesian Deep Learning.

Finally, a probablistic model have the following methods:

-  ``probmodel.summary()``: prints a summary representation of the
   model.
-  ``probmodel.get_config()``: returns a dictionary containing the
   configuration of the model. The model can be reinstantiated from its
   config via:

.. code:: python

    config = probmodel.get_config()
    probmodel = ProbModel.from_config(config)

-  ``model.to_json()``: returns a representation of the model as a JSON
   string. Note that the representation does not include the weights,
   only the architecture. You can reinstantiate the same model (with
   reinitialized weights) from the JSON string via: \`\`\`python from
   models import model\_from\_json

json\_string = model.to\_json() model = model\_from\_json(json\_string)
\`\`\`

--------------

Guide to Approximate Inference in Probabilistic Models
------------------------------------------------------

The API defines the set of algorithms and methods used to perform
inference in a probabilistic model :math:`p(x,z,\theta)` (where
:math:`x` are the observations, :math:`z` the local hidden variibles,
and :math:`\theta` the global parameters of the model). More precisely,
the inference problem reduces to compute the posterior probability over
the latent variables given a data sample
$p(z,:raw-latex:`\theta`\|x\_{train}), because by looking at these
posteriors we can uncover the hidden structure in the data. For the
running example, :math:`p(mu|x_{train})` tells us where the centroids of
the data are, while :math:`p(z_n|x_{train})` shows us to which centroid
every data point belongs to.

InferPy inherits Edward's approach an consider approximate inference
solutions,

.. math::  q(z,\theta) \approx p(z,\theta | x_{train})

,

in which the task is to approximate the posterior
:math:`p(z,\theta | x_{train})` using a family of distributions,
:math:`q(z,\theta; \labmda)`, indexed by a parameter vector
:math:`\lambda`.

A probabilistic model in InferPy should be compiled before we can access
these posteriors,

.. code:: python

     probmodel = ProbModel(vars = [theta,mu,sigma,z_n, x_n]) 
     probmodel.compile(infMethod = 'KLqp')   
     model.fit(x_train)
     posterior_mu = probmodel.posterior(mu)

The compilation process allows to choose the inference algorithm through
the 'infMethod' argument. In the above example we use 'Klqp'. Other
inference algorithms include: 'NUTS', 'MCMC', 'KLpq', etc. Look at ? for
a detailed description of the available inference algorithms.

Following InferPy guiding principles, users can further configure the
inference algorithm.

First, they can define they family 'Q' of approximating distributions,

.. code:: python

     probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
     
     q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
     q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
     q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
     
     probmodel.compile(infMethod = 'KLqp', Q = [q_mu, q_sigma, q_z_n])
     model.fit(x_train)
     posterior_mu = probmodel.posterior(mu)

By default, the posterior **q** belongs to the same distribution family
than **p** , but in the above example we show how we can change that
(e.g. we set the posterior over **mu** to obtain a point mass estimate
instead of the Gaussian approximation used by default). We can also
configure how these **q's** are initialized using any of the Keras's
initializers.

Inspired by Keras semantics, we can furhter configure the inference
algorithm,

.. code:: python

     probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
     
     q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
     q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
     q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
     
     sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
     infkl_qp = inf.inference.KLqp(Q = [q_mu, q_sigma, q_z_n], optimizer = sgd, loss="ELBO")
     probmodel.compile(infMethod = infkl_qp)

     model.fit(x_train)
     posterior_mu = probmodel.posterior(mu)

Have a look at Inference Zoo to explore other configuration options.

In the last part of this guide, we highlight that InferPy directly
builds on top of Edward's compositionality idea to design complex
infererence algorithms.

.. code:: python

     probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 
     
     q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
     q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
     q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
     
     infkl_qp = inf.inference.KLqp(Q = [q_z_n], optimizer = 'sgd', innerIter = 10)
     infMAP = inf.inference.MAP(Q = [q_mu, q_sigma], optimizer = 'sgd')

     probmodel.compile(infMethod = [infkl_qp,infMAP])
     
     model.fit(x_train)
     posterior_mu = probmodel.posterior(mu)

With the above sintaxis, we perform a variational EM algorithm, where
the E step is repeated 10 times for every MAP step.

More flexibility is also available by defining how each mini-batch is
processed by the inference algorithm. The following piece of code is
equivalent to the above one,

.. code:: python

     probmodel = ProbModel(vars = [theta,mu,sigma,z_n,x_n]) 

     q_z_n = inf.inference.Q.Multinomial(bind = z_n, initializer='random_unifrom')
     q_mu = inf.inference.Q.PointMass(bind = mu, initializer='random_unifrom')
     q_sigma = inf.inference.Q.PointMass(bind = sigma, initializer='ones')
     
     infkl_qp = inf.inference.KLqp(Q = [q_z_n])
     infMAP = inf.inference.MAP(Q = [q_mu, q_sigma])

     emAlg = lambda (infMethod, dataBatch):
        for _ in range(10)
            infMethod[0].update(data = dataBatch)
        
        infMethod[1].update(data = dataBatch)
        return 
     
     probmodel.compile(infMethod = [infkl_qp,infMAP], ingAlg = emAlg)
     
     model.fit(x_train, EPOCHS = 10)
     posterior_mu = probmodel.posterior(mu)

Have a look again at Inference Zoo to explore other complex
compositional options.

--------------

Guide to Bayesian Deep Learning
-------------------------------

InferPy inherits Edward's approach for representing probabilistic models
as (stochastic) computational graphs. As describe above, a random
variable :math:`x` is associated to a tensor :math:`x^*` in the
computational graph handled by TensorFlow, where the computations takes
place. This tensor :math:`x^*` contains the samples of the random
variable :math:`x`, i.e. :math:`x^* \sim p(x|\theta)`. In this way,
random variables can be involved in complex deterministic operations
containing deep neural networks, math operations and another libraries
compatible with Tensorflow (such as Keras).

Bayesian deep learning or deep probabilistic programming enbraces the
idea of employing deep neural networks within a probabilistic model in
order to capture complex non-linear dependencies between variables.

InferPy's API gives support to this powerful and flexible modelling
framework. Let us start by showing how a variational autoencoder over
binary data can be defined by mixing Keras and InferPy code.

.. code:: python

    from keras.models import Sequential
    from keras.layers import Dense, Activation

    M = 1000
    dim_z = 10
    dim_x = 100

    #Define the decoder network
    input_z  = keras.layers.Input(input_dim = dim_z)
    layer = keras.layers.Dense(256, activation = 'relu')(input_z)
    output_x = keras.layers.Dense(dim_x)(layer)
    decoder_nn = keras.models.Model(inputs = input, outputs = output_x)

    #define the generative model
    with inf.replicate(size = N)
     z = Normal(0,1, dim = dim_z)
     x = Bernoulli(logits = decoder_nn(z.value()), observed = true)

    #define the encoder network
    input_x  = keras.layers.Input(input_dim = d_x)
    layer = keras.layers.Dense(256, activation = 'relu')(input_x)
    output_loc = keras.layers.Dense(dim_z)(layer)
    output_scale = keras.layers.Dense(dim_z, activation = 'softplus')(layer)
    encoder_loc = keras.models.Model(inputs = input, outputs = output_mu)
    encoder_scale = keras.models.Model(inputs = input, outputs = output_scale)

    #define the Q distribution
    q_z = Normal(loc = encoder_loc(x.value()), scale = encoder_scale(x.value()))

    #compile and fit the model with training data
    probmodel.compile(infMethod = 'KLqp', Q = {z : q_z})
    probmodel.fit(x_train)

    #extract the hidden representation from a set of observations
    hidden_encoding = probmodel.predict(x_pred, targetvar = z)

In this case, the parameters of the encoder and decoder neural networks
are automatically managed by Keras. These parameters are them treated as
model parameters and not exposed to the user. In consequence, we can not
be Bayesian about them by defining specific prior distributions. In this
example (?) , we show how we can avoid that by introducing extra
complexity in the code.

Other examples of probabilisitc models using deep neural networks are: -
Bayesian Neural Networks - Mixture Density Networks - ...

We can also define a Keras model whose input is an observation and its
output its the expected value of the posterior over the hidden
variables, :math:`E[p(z|x)]`, by using the method 'toKeras', as a way to
create more expressive models.

.. code:: python

    from keras.layers import Conv2D, MaxPooling2D, Flatten
    from keras.layers import Input, LSTM, Embedding, Dense
    from keras.models import Model, Sequential

    #We define a Keras' model whose input is data sample 'x' and the output is the encoded vector E[p(z|x)]
    variational_econder_keras = probmodel.toKeras(targetvar = z)

    vision_model = Sequential()
    vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    vision_model.add(Conv2D(64, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    # Now let's get a tensor with the output of our vision model:
    encoded_image = vision_model(input_x)

    # Let's concatenate the vae vector and the convolutional image vector:
    merged = keras.layers.concatenate([variational_econder_keras, encoded_image])

    # And let's train a logistic regression over 100 categories on top:
    output = Dense(100, activation='softmax')(merged)

    # This is our final model:
    classifier = Model(inputs=[input_x], outputs=output)

    # The next stage would be training this model on actual data.

Guide to Validation of Probabilistic Models
-------------------------------------------

Model validation try to assess how faifhfully the inferered
probabilistic model represents and explain the observed data.

The main tool for model validation consists on analyzing the posterior
predictive distribution,

:math:`p(y_{test}, x_{test}|y_{train}, x_{train}) = \int p(y_{test}, x_{test}|z,\theta)p(z,\theta|y_{train}, x_{train}) dzd\theta`


This posterior predictive distribution can be used to measure how well
the model fits an independent dataset using the test marginal
log-likelihood, :math:`\ln p(y_{test}, x_{test}|y_{train}, x_{train})`,

.. code:: python

    log_like = probmodel.evaluate(test_data, metrics = ['log_likelihood'])

In other cases, we may need to evalute the predictive capacity of the
model with respect to some target variable :math:`y`,

:math:`p(y_{test}|x_{test}, y_{train}, x_{train}) = \int p(y_{test}|x_{test},z,\theta)p(z,\theta|y_{train}, x_{train}) dzd\theta`

So the metrics can be computed with respect to this target variable by
using the ‘targetvar’ argument,

.. code:: python

    log_like, accuracy, mse = probmodel.evaluate(test_data, targetvar = y, metrics = ['log_likelihood', 'accuracy', 'mse'])

So, the log-likelihood metric as well as the accuracy and the mean
square error metric are computed by using the predictive posterior
:math:`p(y_{test}|x_{test}, y_{train}, x_{train})`.

Custom evaluation metrics can also be defined,

.. code:: python

    def mean_absolute_error(posterior, observations, weights=None):
        predictions = tf.map_fn(lambda x : x.getMean(), posterior)
        return tf.metrics.mean_absolute_error(observations, predictions, weights)
        
    mse, mae = probmodel.evaluate(test_data, targetvar = y, metrics = ['mse', mean_absolute_error])

--------------



Guide to Data Handling
----------------------

.. code:: python

    import numpy as np
    import inferpy as inf
    from inferpy.models import Normal, InverseGamma, Dirichlet

    #We first define the probabilistic model 
    with inf.ProbModel() as mixture_model:
        # K defines the number of components. 
        K=10
        #Prior for the means of the Gaussians 
        mu = Normal(loc = 0, scale = 1, shape=[K,d])
        #Prior for the precision of the Gaussians 
        invgamma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
        #Prior for the mixing proportions
        theta = Dirichlet(np.ones(K))

        # Number of observations
        N = 1000
        #data Model
        with inf.replicate(size = N, batch_size = 100)
            # Sample the component indicator of the mixture. This is a latent variable that can not be observed
            z_n = Multinomial(probs = theta)
            # Sample the observed value from the Gaussian of the selected component.  
            x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(invgamma,z_n), observed = true)

    #compile the probabilistic model
    mixture_model.compile(infAlg = 'klqp')

    #fit the model with data
    mixture_model.fit(data)

--------------



