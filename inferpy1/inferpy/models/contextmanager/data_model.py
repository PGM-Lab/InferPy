from contextlib import contextmanager
from . import prob_model


# This dict store the active (if active is True) context parameters of datamodel
_active_datamodel = dict(
    sample_shape=None,
    active=False
)


def is_active():
    # is the prob model builder context active?
    return _active_datamodel['active']


def _is_connected_to_datamodel_var(name):
    graph = prob_model.get_graph()
    # is this a Random Variable with any child being in a data model?
    # NOTE: we use the model variables because children (successors) is_datamodel attribute is built initially
    return any(prob_model.get_model_variable(pname).is_datamodel for pname in graph.successors(name))


def _is_expanded_var_parameters(name):
    graph = prob_model.get_graph()
    # is this a Random Variable with any parent expanded? If any, return True (will be expanded by parent sample_shape)
    # NOTE: we use the builder variables because parents (predecessors) is_expanded attribute is built right now
    return any(prob_model.get_builder_variable(pname).is_expanded for pname in graph.predecessors(name))


def get_random_variable_shape(var_args, var_kwargs):
    # 1)
    # In this context, we have to expand only if variable is in a datamodel context.
    # If var parameters are not expanded, and sample_shape has been provided, then expand.
    # 2)
    # If in prob model context, a variable out of datamodel context migh need to be broadcasted. This happens when
    # the variable (parent) is connected to a random variable in a data model context (child).
    #
    # Return a dict with keys:
    # sample_shape: The sample shape (a tuple, an integer, a Tensor or a TensorShape)
    # is_expanded: Is this variable expandedndirectly (by sample_shape) or indirectly
    #              (by broadcasting because of any of its parents, that is, var parameters)?
    # broadcast_shape: The shape of the broadcast_to operation if needs to be applied

    # 1)
    if prob_model.is_active() and _active_datamodel['active'] and _active_datamodel['sample_shape'] is not None:
        # we need to expand this variable.
        is_expanded = True
        # Parameters already expanded?
        # In probmodel definitions, each RandomVariable must have a name
        if _is_expanded_var_parameters(var_kwargs['name']):
            # yes, do not need to expand this var (it will be expanded by broadcast)
            sample_shape = ()
        else:
            # no, we need to expand this variable
            sample_shape = _active_datamodel['sample_shape']
    else:
        # not need to expand this variable
        sample_shape, is_expanded = (), False

    # 2)
    broadcast_shape = ()
    if prob_model.is_active() and not _active_datamodel['active'] and _active_datamodel['sample_shape'] is not None \
            and _is_connected_to_datamodel_var(var_kwargs['name']):
            # inside prob model, not inside a data model, sample_shape is not None and
            # is directly connected to a random variable inside a data model
            if hasattr(_active_datamodel['sample_shape'], '__iter__'):
                # if sample_shape is iterable, asign directly
                broadcast_shape = _active_datamodel['sample_shape']
            else:
                # otherwise, make it a tuple with one element, so broadcast_shape is always iterable
                broadcast_shape = (_active_datamodel['sample_shape'], )

    return dict(
        sample_shape=sample_shape,
        is_expanded=is_expanded,
        broadcast_shape=broadcast_shape
    )


@contextmanager
def fit(sample_shape):
    # Fit the datamodel parameters. We only allow to use one context level
    assert _active_datamodel['sample_shape'] is None
    _active_datamodel['sample_shape'] = sample_shape

    try:
        yield
    finally:
        _active_datamodel['sample_shape'] = None


@contextmanager
def datamodel():
    # Context decorator. A function with no parameters. We only allow to use one context level
    assert not _active_datamodel['active']
    _active_datamodel['active'] = True
    try:
        yield
    finally:
        _active_datamodel['active'] = False
