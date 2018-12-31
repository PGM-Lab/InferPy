import inferpy as inf

def test_random_variable_simple():
    name = "x"

    x = inf.models.Normal(loc=1., scale=100, name=name)
    assert x.shape == 1
    assert x.name == name
