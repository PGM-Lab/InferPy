from contextlib import contextmanager, ExitStack
from . import randvar_registry
from inferpy import exceptions


# This dict store the active (if active is True) context and the size of the dtamodel plate in the current context
_active_datamodel = dict(
    size=1,
    active=False
)


def is_active():
    # is the data model context active?
    return _active_datamodel['active']


def _has_datamodel_var_parameters(name):
    graph = randvar_registry.get_graph()
    # is this a Random Variable with any parent expanded? If any, return True (will be expanded by parent size)
    # NOTE: we use the builder variables because parents (predecessors) is_datamodel attribute is built right now
    return any(randvar_registry.get_variable(pname).is_datamodel for pname in graph.predecessors(name))


def get_sample_shape(name):
    # In this context, we have to expand only if the element is in a datamodel context.
    # Assert that element is in a model and datamodel is active
    # If var parameters are not expanded, and size has been provided, then expand.
    #
    # Return a the sample_shape (number of samples of the datamodel). It is an integer, or ().

    # Parameters already expanded?
    # In probmodel definitions, each RandomVariable must have a name
    if _has_datamodel_var_parameters(name):
        # yes, do not need to expand this var (it will be expanded by broadcast)
        size = ()
    else:
        # no, we need to expand this variable
        size = _active_datamodel['size']

    return size


@contextmanager
def fit(size):
    # size must be an integer
    if not isinstance(size, int):
        raise exceptions.NotIntegerDataModelSize(
            'The size of the data model must be an integer, not : {}'.format(type(size)))
    # Fit the datamodel parameters. We only allow to use one context level
    assert _active_datamodel['size'] == 1
    _active_datamodel['size'] = size

    try:
        yield
    finally:
        _active_datamodel['size'] = 1


@contextmanager
def datamodel(size=1):
    # Context decorator. We only allow to use one context level
    assert not _active_datamodel['active']
    _active_datamodel['active'] = True

    # collect all the contextmanager decorated functions that we want to use before yield
    contexts = []
    # if size is provided and greater to 1, use the fit context
    if size > 1:
        contexts.append(fit(size))

    # use the ExitStack to enter the context if exists, or do nothing if contexts is empty
    try:
        with ExitStack() as stack:
            for c in contexts:
                stack.enter_context(c)
            yield
    finally:
        _active_datamodel['active'] = False
