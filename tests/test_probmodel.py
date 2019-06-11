import numpy as np
import pytest

import inferpy as inf
from inferpy import util
from tests import no_raised_exc


def test_build_model():
    sample_size = 100
    batch_shape = (2, 3)

    @inf.probmodel
    def model():
        p = inf.Parameter(np.zeros(batch_shape, dtype=np.float32), name='p')
        with inf.datamodel():
            x = inf.Normal(p, 1., name='x')
            inf.Normal(x, 1., name='y')

    m = model()
    # builder is a function
    assert callable(m.builder)
    # vars is an OrderedDict with the random variables
    assert 'x' in m.vars
    assert 'y' in m.vars
    # params is an OrderedDict with the parameters
    assert 'p' in m.params

    # if fit for datamodel not used, then the shape of random vars is (1, batch_shape)
    assert tuple(m.vars['x'].shape.as_list()) == (1, ) + batch_shape
    assert tuple(m.vars['y'].shape.as_list()) == (1, ) + batch_shape
    # if expanded datamodel, then the shape of random vars is (sample_shape, batch_shape)
    expanded_vars, _ = m.expand_model(sample_size)
    assert tuple(expanded_vars['x'].shape.as_list()) == (sample_size, ) + batch_shape
    assert tuple(expanded_vars['y'].shape.as_list()) == (sample_size, ) + batch_shape

    # assert variables and parameters are in graph
    assert 'x' in m.graph
    assert 'y' in m.graph
    assert 'p' in m.graph


@pytest.mark.parametrize("data, expected_flow, expected_result", [
    # empty dict
    (
        dict(),
        no_raised_exc(),
        1
    ),
    # one field which does not exist
    (
        dict(othername=np.ones(100)),
        no_raised_exc(),
        1
    ),
    # one variable data size
    (
        dict(x=np.ones(100)),
        no_raised_exc(),
        100
    ),
    # two variable data with the same size
    (
        dict(x=np.ones(100), y=np.ones(100)),
        no_raised_exc(),
        100
    ),
    # two variable data with different size
    (
        dict(x=np.ones(100), y=np.ones(150)),
        pytest.raises(ValueError),
        None
    ),
])
def test_probmodel_get_plate_size(data, expected_flow, expected_result):
    @inf.probmodel
    def model():
        p = inf.Parameter(0., name='p')
        with inf.datamodel():
            x = inf.Normal(p, 1., name='x')
            inf.Normal(x, 1., name='y')

    with expected_flow:
        m = model()

        plate_size = util.iterables.get_plate_size(m.vars, data)
        assert expected_result == plate_size


def test_sample():
    @inf.probmodel
    def model():
        p = inf.Parameter(0., name='p')
        with inf.datamodel():
            x = inf.Normal(p, 1., name='x')
            inf.Normal(x, 1., name='y')

    N = 100
    m = model()

    sample_dict = m.prior().sample(N)

    varnames = list(sample_dict.keys())
    assert len(varnames) == 2
    assert 'x' in varnames
    assert 'y' in varnames

    assert len(sample_dict['x']) == N
    assert len(sample_dict['y']) == N


def test_sample_intercept():
    @inf.probmodel
    def model():
        p = inf.Parameter(0., name='p')
        with inf.datamodel():
            x = inf.Normal(p, 1., name='x')
            inf.Normal(x, 1., name='y')

    N = 10
    data_y = 1.0
    m = model()

    sample_dict = m.prior(data={'y': data_y}).sample(N)

    varnames = list(sample_dict.keys())
    assert len(varnames) == 2
    assert 'x' in varnames
    assert 'y' in varnames

    assert len(sample_dict['x']) == N
    assert len(sample_dict['y']) == N


def test_log_prob():
    @inf.probmodel
    def model():
        p = inf.Parameter(0., name='p')
        with inf.datamodel():
            x = inf.Normal(p, 1., name='x')
            inf.Normal(x, 1., name='y')

    m = model()

    data = m.prior(['x', 'y']).sample()
    print(data)
    logprob_dict = m.prior(['x', 'y']).log_prob()
    varnames = list(logprob_dict.keys())
    assert len(varnames) == 2
    assert 'x' in varnames
    assert 'y' in varnames

    assert logprob_dict['x'] <= 0.0
    assert logprob_dict['y'] <= 0.0

    # assert that the result of sum_log_prob is a single float32 number
    assert isinstance(m.prior(data=data).sum_log_prob(), np.float32)
