Guide to Bayesian Deep Learning
===============================

.. note:: not implemented yet

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

InferPy's API gives support to this powerful and flexible modeling
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
variables, :math:`E[p(z|x)]`, by using the method ``toKeras``, as a way to
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
    

