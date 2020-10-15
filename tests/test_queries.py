import numpy as np
import pytest
import tensorflow as tf

import inferpy as inf
from inferpy import util
from tests import no_raised_exc


def init_model():
    @inf.probmodel
    def linear_reg(d):
        w0 = inf.Normal(0, 1, name="w0")
        w = inf.Normal(np.zeros([d, 1]), 1, name="w")

        with inf.datamodel():
            x = inf.Normal(tf.ones(d), 2, name="x")
            y = inf.Normal(w0 + x @ w, 1.0, name="y")


    @inf.probmodel
    def qmodel(d):
        qw0_loc = inf.Parameter(0., name="qw0_loc")
        qw0_scale = tf.math.softplus(inf.Parameter(1., name="qw0_scale"))
        qw0 = inf.Normal(qw0_loc, qw0_scale, name="w0")

        qw_loc = inf.Parameter(np.zeros([d, 1]), name="qw_loc")
        qw_scale = tf.math.softplus(inf.Parameter(tf.ones([d, 1]), name="qw_scale"))
        qw = inf.Normal(qw_loc, qw_scale, name="w")


    # create an instance of the model
    m = linear_reg(d=2)
    q = qmodel(2)
    # create toy data
    N = 1000
    data = m.prior(["x", "y"], data={"w0": 0, "w": [[2], [1]]}, size_datamodel=N).sample()


    x_train = data["x"]
    y_train = data["y"]

    # set and run the inference
    VI = inf.inference.VI(qmodel(2), epochs=500)
    m.fit({"x": x_train, "y": y_train}, VI)

    return m, x_train, y_train



def test_prior():
    m, x_train, y_train = init_model()

    s = m.prior().sample()
    # l = m.prior().log_prob(s)

    print({k: v.shape for k, v in s.items()})

    assert s["w0"].shape == ()
    assert s["w"].shape == (2, 1)
    assert s["x"].shape == (1, 2)
    assert s["y"].shape == (1, 1)

    # assert l["w0"].shape == ()
    # assert l["w"].shape == (2, 1)
    # assert l["x"].shape == (1, 2)
    # assert l["y"].shape == (1, 1)


def test_prior_size():
    m, x_train, y_train = init_model()
    s = m.prior().sample(size=5)

    print({k: v.shape for k, v in s.items()})

    assert s["w0"].shape == (5,)
    assert s["w"].shape == (5, 2, 1)
    assert s["x"].shape == (5, 1, 2)
    assert s["y"].shape == (5, 1, 1)


def test_prior_size_datamodel():
    m, x_train, y_train = init_model()
    s = m.prior(size_datamodel=10).sample()

    print({k: v.shape for k, v in s.items()})

    assert s["w0"].shape == ()
    assert s["w"].shape == (2, 1)
    assert s["x"].shape == (10, 2)
    assert s["y"].shape == (10, 1)

def test_prior_size2():
    m, x_train, y_train = init_model()
    s = m.prior(size_datamodel=10).sample(size=5)

    print({k: v.shape for k, v in s.items()})

    assert s["w0"].shape == (5,)
    assert s["w"].shape == (5, 2, 1)
    assert s["x"].shape == (5, 10, 2)
    assert s["y"].shape == (5, 10, 1)


def test_prior_data():
    m, x_train, y_train = init_model()
    s = m.prior(size_datamodel=10, data={"w0": 0, "w": [[2], [1]], "y": np.zeros((10, 1))}).sample(size=5)

    print({k: v.shape for k, v in s.items()})

    assert s["w0"].shape == (5,)
    assert s["w"].shape == (5, 2, 1)
    assert s["x"].shape == (5, 10, 2)
    assert s["y"].shape == (5, 10, 1)


def test_prior_batches():
    m, x_train, y_train = init_model()
    s = m.prior(size_datamodel=10, data={"w0": 0, "w": [[2], [1]], "y": np.zeros((15, 1))}).sample(size=5)  # ERROR

    print({k: v.shape for k, v in s.items()})

    assert s["w0"].shape == (5,)
    assert s["w"].shape == (5, 2, 1)
    assert s["x"].shape == (5, 15, 2)
    assert s["y"].shape == (5, 15, 1)


def test_post():
    m, x_train, y_train = init_model()
    s = m.posterior_predictive().sample()

    print({k: v.shape for k, v in s.items()})

    assert s["x"].shape == (1000, 2)
    assert s["y"].shape == (1000, 1)


def test_post_size():
    m, x_train, y_train = init_model()
    s = m.posterior_predictive().sample(size=5)

    print({k: v.shape for k, v in s.items()})

    assert s["x"].shape == (5, 1000, 2)
    assert s["y"].shape == (5, 1000, 1)


def test_post_data():
    m, x_train, y_train = init_model()
    s = m.posterior_predictive(data={"x":x_train}).sample(3)

    print({k:v.shape for k,v in s.items()})

    assert s["x"].shape == (3,1000,2)
    assert s["y"].shape == (3,1000,1)



def test_post_data2():
    m, x_train, y_train = init_model()
    s = m.posterior_predictive(data={"w0": 0, "w": [[2], [1]], "x": np.zeros((1000, 2))}).sample(3)

    print({k: v.shape for k, v in s.items()})

    assert s["x"].shape == (3, 1000, 2)
    assert s["y"].shape == (3, 1000, 1)


def test_post_batches():
    m, x_train, y_train = init_model()
    s = m.posterior_predictive(data={"y": np.zeros((1300, 1))}).sample(3)  # ERROR

    print({k: v.shape for k, v in s.items()})

    assert s["x"].shape == (3, 1300, 2)
    assert s["y"].shape == (3, 1300, 1)







