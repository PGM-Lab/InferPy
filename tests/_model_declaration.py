import inferpy as inf
import numpy as np


# TODO: Design multiple unit test functions. This is more like a functional test rather than unit test.
def test_model():
    with inf.ProbModel() as m:
        x = inf.models.Normal(loc=1., scale=100, name="x")

        with inf.replicate(size=100):
            y = inf.models.Normal(loc=x, scale=0.0001, dim=3, name="y", observed=True)

    # print the list of variables
    print(m.varlist)
    print(m.latent_vars)
    print(m.observed_vars)

    # get a sample
    m_sample = m.sample()
    print("sample:")
    print(m_sample)

    assert np.abs(np.mean(list(m_sample.values())[0] - list(m_sample.values())[1])) < 1

    # compute the log_prob for each element in the sample
    print(m.log_prob(m_sample))

    # compute the sum of the log_prob
    print(m.sum_log_prob(m_sample))

    assert len(m.varlist) == 2
    assert len(m.latent_vars) == 1
    assert len(m.latent_vars) == 1

    assert m.is_compiled()

    m.compile()

    assert m.is_compiled()

    z = inf.models.Normal(loc=1., scale=1., dim=3, name="z")
    m.add_var(z)

    assert not m.is_compiled()
