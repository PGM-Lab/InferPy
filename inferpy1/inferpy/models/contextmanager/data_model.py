from contextlib import contextmanager
from . import prob_model


# This dict store the active (if active is True) context parameters of datamodel
_active_datamodel = dict(
    size=1,
    active=False
)


def is_active():
    # is the prob model builder context active?
    return _active_datamodel['active']


def _is_expanded_var_parameters(name):
    graph = prob_model.get_graph()
    # is this a Random Variable with any parent expanded? If any, return True (will be expanded by parent size)
    # NOTE: we use the builder variables because parents (predecessors) is_expanded attribute is built right now
    return any(prob_model.get_builder_variable(pname).is_expanded for pname in graph.predecessors(name))


def get_random_variable_shape(var_args, var_kwargs):
    # In this context, we have to expand only if variable is in a datamodel context.
    # If var parameters are not expanded, and size has been provided, then expand.
    #
    # Return a tuple with elements:
    # size: The number of samples of the datamodel (an integer)
    # is_expanded: Is this variable expanded directly (by size) or indirectly
    #              (by broadcasting because of any of its parents, that is, var parameters)?

    if prob_model.is_active() and _active_datamodel['active']:
        # we need to expand this variable.
        is_expanded = True
        # Parameters already expanded?
        # In probmodel definitions, each RandomVariable must have a name
        if _is_expanded_var_parameters(var_kwargs['name']):
            # yes, do not need to expand this var (it will be expanded by broadcast)
            size = ()
        else:
            # no, we need to expand this variable
            size = _active_datamodel['size']
    else:
        # not need to expand this variable
        size, is_expanded = (), False

    return size, is_expanded


@contextmanager
def fit(size):
    # Fit the datamodel parameters. We only allow to use one context level
    assert _active_datamodel['size'] == 1
    _active_datamodel['size'] = size

    try:
        yield
    finally:
        _active_datamodel['size'] = 1


@contextmanager
def datamodel():
    # Context decorator. A function with no parameters. We only allow to use one context level
    assert not _active_datamodel['active']
    _active_datamodel['active'] = True
    try:
        yield
    finally:
        _active_datamodel['active'] = False
