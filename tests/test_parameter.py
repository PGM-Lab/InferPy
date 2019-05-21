import inferpy as inf


def test_parameter_in_pmodel():
    # test that random variables in pmodel works even if no name has been provided
    @inf.probmodel
    def model():
        inf.Parameter(0)

    v = list(model().params.values())[0]
    assert v.name.startswith('parameter')
    # assert also is_datamodel is false
    assert not v.is_datamodel


def test_parameter_in_datamodel():
    with inf.datamodel(10):
        x = inf.Parameter(0)

    # assert that is_datamodel is true
    assert x.is_datamodel


def test_run_in_session():
    x = inf.Parameter(0)
    assert inf.get_session().run(x) == 0
