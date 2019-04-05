import pytest

import inferpy as inf
from inferpy.contextmanager import randvar_registry
from inferpy.util import tf_graph


@pytest.fixture(params=[
    # default context, build graph
    (True, True),
    # new context, build graph
    (False, True),
    # new context, not build graph
    (False, False)
])
def init_context(request):
    is_default, is_building = request.param
    result = {
        'is_building': is_building,
        'is_default': is_default
    }
    if not is_default:
        with randvar_registry.init(None if is_building else tf_graph.get_empty_graph()):
            yield result
    else:
        yield result


def test_context(init_context):
    assert randvar_registry.is_building_graph() is init_context['is_building']
    assert randvar_registry.is_default() is init_context['is_default']


def test_register_variable(init_context):
    is_default = init_context['is_default']
    name = 'x'

    # the variable does not exist in the context
    assert randvar_registry.get_variable(name) is None
    assert randvar_registry.get_variable_or_parameter(name) is None
    inf.Normal(0, 1, name=name)

    # the variable exists in the context
    # randvar_registry.register_variable(x) has been automatically called
    assert randvar_registry.get_variable(name) is not None
    assert randvar_registry.get_variable_or_parameter(name) is not None

    # if create a new variable with the same name, it fails just if is_default is False
    if is_default:
        inf.Normal(0, 1, name=name)
        assert randvar_registry.get_variable(name) is not None
        assert randvar_registry.get_variable_or_parameter(name) is not None
    else:
        with pytest.raises(ValueError):
            inf.Normal(0, 1, name=name)


def test_register_parameter(init_context):
    is_default = init_context['is_default']
    name = 'x'

    # the parameter does not exist in the context
    assert len(randvar_registry.get_var_parameters()) == 0
    assert randvar_registry.get_variable_or_parameter(name) is None
    p = inf.Parameter(0, name=name)

    # the parameter exists in the context
    # randvar_registry.register_variable(x) has been automatically called
    assert len(randvar_registry.get_var_parameters()) == 1
    assert randvar_registry.get_var_parameters()[name] == p
    assert randvar_registry.get_variable_or_parameter(name) == p

    # if create a new parameter with the same name, it fails just if is_default is False
    if is_default:
        p = inf.Parameter(0, name=name)
        assert len(randvar_registry.get_var_parameters()) == 1
        assert randvar_registry.get_var_parameters()[name] == p
        assert randvar_registry.get_variable_or_parameter(name) == p
    else:
        with pytest.raises(ValueError):
            inf.Parameter(0, name=name)


def test_graph(init_context):
    # this test check if calling to update_graph update the graph or not depending on is_building
    is_building = init_context['is_building']
    elems = []  # we use this variable to append the expected elements in the graph

    # the variable does not exist in the context
    assert len(randvar_registry.get_graph()) == len(elems)
    inf.Normal(0, 1, name='x')
    randvar_registry.update_graph()
    if is_building:
        elems.append('x')

    # the variable exists in the context
    # randvar_registry.register_variable(x) has been automatically called
    assert len(randvar_registry.get_graph()) == len(elems)
    for elem in elems:
        assert elem in randvar_registry.get_graph()

    inf.Normal(0, 1, name='y')
    randvar_registry.update_graph()
    if is_building:
        elems.append('y')

    # the variable exists in the context
    # randvar_registry.register_variable(x) has been automatically called
    assert len(randvar_registry.get_graph()) == len(elems)
    for elem in elems:
        assert elem in randvar_registry.get_graph()
