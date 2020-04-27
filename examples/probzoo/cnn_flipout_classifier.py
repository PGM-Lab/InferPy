import tensorflow as tf
import inferpy as inf
from inferpy.data import mnist
import tensorflow_probability as tfp
import numpy as np

N = 1000 # data size
(x_train, y_train), (x_test, y_test) = mnist.load_data(num_instances=N,
                                  digits=[0, 1], vectorize=False)

S = np.shape(x_train)[1:]


@inf.probmodel
def cnn_flipout_classifier(S):
    with inf.datamodel():
        x = inf.Normal(tf.ones(S), 1, name="x")

        nn = inf.layers.Sequential([
            tfp.layers.Convolution2DFlipout(4, kernel_size=(10,10), padding="same", activation="relu"),
            tf.keras.layers.GlobalMaxPool2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])


        y = inf.Normal(nn(tf.expand_dims(x, 1)), 0.001, name="y")



p = cnn_flipout_classifier(S)


# Empty Q model
@inf.probmodel
def qmodel():
    pass
q = qmodel()

# set the inference algorithm
VI = inf.inference.VI(q, epochs=2000)

# learn the parameters
p.fit({"x": x_train, "y":y_train}, VI)


# evaluate the model

def evaluate(p, x, y):
    N = np.shape(x)[0]
    output = p.posterior_predictive("y", {"x": x[:N]}).sample()
    x_pred = np.reshape(1* (output>0.5), (N,))
    return np.sum(x_pred == y)/N

acc = evaluate(p, x_test[:N], y_test[:N])
print(f"accuracy = {acc}")




