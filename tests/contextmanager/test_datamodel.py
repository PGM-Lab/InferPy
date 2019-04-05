import pytest

import inferpy as inf
from inferpy.contextmanager import data_model


@pytest.mark.parametrize("size", [
    1,
    10,
    1000
])
def test_datamodel(size):
    with data_model.datamodel(size):
        x = inf.Normal(0, 1, name='x')
    assert x.sample_shape == size


@pytest.mark.parametrize("size", [
    1,
    10,
    1000
])
def test_fit(size):
    with data_model.fit(size):
        with data_model.datamodel():
            x = inf.Normal(0, 1, name='x')
    assert x.sample_shape == size


def test_is_active():
    assert not data_model.is_active()
    with data_model.datamodel():
        assert data_model.is_active()
    assert not data_model.is_active()


def test_has_datamodel_var_parameter():
    x = inf.Normal(0, 1)
    with data_model.datamodel(size=10):
        y = inf.Normal(x, 1)
        z = inf.Normal(y, 1)

    # uses the default random variable registry
    assert not data_model._has_datamodel_var_parameters(x.name)  # outside datamodel
    assert not data_model._has_datamodel_var_parameters(y.name)  # first level in datamodel (not var param)
    assert data_model._has_datamodel_var_parameters(z.name)  # has the y rand var as parameter
