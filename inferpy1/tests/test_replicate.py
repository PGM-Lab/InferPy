import inferpy as inf
import tensorflow as tf
import pytest


@pytest.fixture
def reset_replicate():

    tf.reset_default_graph()

    # delete existing replicate
    inf.replicate.delete_all()

    yield


@pytest.fixture
def toy_model_replicate():

    h0 = inf.models.Normal(0,1, observed=False)

    with inf.replicate(size=10, name="A") as A:
        h10 = inf.models.Normal(0, 1, name="h10", observed=False)
        h11 = inf.models.Normal(0, 1, name="h11", observed=False)
        x10 = inf.models.Normal(0, 1, name="x10", observed=True)

        with inf.replicate(size=5, name="B") as B:
            h20 = inf.models.Normal(0, 1, name="h20", observed=False)
            h21 = inf.models.Normal(0, 1, name="h21", observed=False)
            x20 = inf.models.Normal(0, 1, name="x20", observed=True)


    return A, B


## function that asserts the main static functions of replicate ###

def assert_replicate_state(in_rep, num_active, num_all, total_size):
    assert inf.replicate.in_replicate() == in_rep
    assert len(inf.replicate.get_active_replicate()) == num_active
    assert len(inf.replicate.get_all_replicate()) == num_all
    assert inf.replicate.get_total_size() == total_size



##
# Set of tests checking the static functions with at multiple settings
###


# when there is not any defined replicate construct
def test_no_replicate(reset_replicate):
    assert_replicate_state(in_rep = False, num_active=0, num_all=0, total_size = 1)


# inside a simple replicate construct
def test_single_replicate(reset_replicate):
    with inf.replicate(size=5):
        assert_replicate_state(in_rep=True, num_active=1, num_all=1, total_size=5)


# outside a simple reflicate construct
def test_after_replicate(reset_replicate):
    with inf.replicate(size=5):
        pass
    assert_replicate_state(in_rep=False, num_active=0, num_all=1, total_size=1)



# two nested replicate constructs
def test_nested_replicate(reset_replicate):
    with inf.replicate(size=5):
        with inf.replicate(size=10):
            assert_replicate_state(in_rep=True, num_active=2, num_all=2, total_size=50)



# re-usage of a previously defined replicate construct
def test_reused_replicate(reset_replicate):
    with inf.replicate(size=6, name = "A"):
        pass

    with inf.replicate(name="A"):
        assert_replicate_state(in_rep=True, num_active=1, num_all=1, total_size=6)


# re-usage of 2 previously defined replicate constructs
def test_compund_replicate(reset_replicate):
    with inf.replicate(size=6, name="A"):
        pass
    with inf.replicate(size=5, name="B"):
        pass

    with inf.replicate(name="A"):
        with inf.replicate(name="B"):
            assert_replicate_state(in_rep=True, num_active=2, num_all=2, total_size=30)


# delete all (active and non-active) constructs
def test_delete_replicate(reset_replicate):
    with inf.replicate(size=4):
        pass

    inf.replicate.delete_all()

    assert_replicate_state(in_rep = False, num_active=0, num_all=0, total_size = 1)




# check the list of variables inside a replicate
def test_varlist(reset_replicate, toy_model_replicate):

    A,B = toy_model_replicate


    pass


@fixture_reset_replicate
@fixture_toy_model
def test_get_replicate_by_name():
    pass





# note: local_hidden functionality in random variable tests







