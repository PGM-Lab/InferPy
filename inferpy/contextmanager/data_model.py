from contextlib import contextmanager, ExitStack
from . import randvar_registry


# This dict store the active (if active is True) context and the size of the dtamodel plate in the current context
# The datamodel means that the variables inside are expanded. At least, the size is equals 1.
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
    # To do that we check the is_datamodel from the parents (predecessors), which can be variables or parameters
    return any(randvar_registry.get_variable_or_parameter(pname).is_datamodel for pname in graph.predecessors(name))


def get_sample_shape(name):
    """
    This function must be used inside a datamodel context (it is not checked here)
    If var parameters are not expanded, then expand.
        :name (str): The name of the variable to get its sample shape
        :returns: a the sample_shape (number of samples of the datamodel). It is an integer, or ().
    """

    # Parameters already expanded? (remember that in probmodel definitions, each RandomVariable must have a name)
    if _has_datamodel_var_parameters(name):
        # yes, do not need to expand this var (it will be expanded by parents)
        size = ()
    else:
        # no, we need to expand this variable
        size = _active_datamodel['size']

    return size


@contextmanager
def fit(size):
    # size must be an integer
    if not isinstance(size, int):
        raise TypeError('The size of the data model must be an integer, not : {}'.format(type(size)))
    # Fit the datamodel parameters
    _active_datamodel['size'] = size

    try:
        yield
    finally:
        _active_datamodel['size'] = 1


@contextmanager
def datamodel(size=None):
    """
    This context is used to declare a plateau model. Random Variables and Parameters will use a sample_shape
    defined by the argument `size`, or by the `data_model.fit`. If `size` is not specify, the default size 1,
    or the size specified by `fit` will be used.
    """

    # NOTE: We only allow to use one context level, assert that it is not active now
    assert not _active_datamodel['active']
    _active_datamodel['active'] = True

    # to simplify the code, avoiding if-else blocks, we declare a list of contexts (empty or with one fit if `size`)
    # and use the ExitStack() to enter all of them at the same time (if empty, it does nothing)
    contexts = []
    # if size is provided, use the fit context
    if size:
        contexts.append(fit(size))

    # use the ExitStack to enter the context, or do nothing if contexts is empty
    try:
        with ExitStack() as stack:
            for c in contexts:
                stack.enter_context(c)
            yield
    finally:
        _active_datamodel['active'] = False
