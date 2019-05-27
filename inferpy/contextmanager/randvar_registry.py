from contextlib import contextmanager
from inferpy.util import tf_graph
import warnings


# This dict store the context parameters of random variable and parameters registry; if the graph is being built or not;
# the graph of dependencies (can be built here or provided through the init function), and the builder vars and params
# that are being built now.

# Finally, there is a special registry which is the default (marked with `is_default`). The default registry is used to
# declare variables and parameters just to play, and therefore it is allowed to declare these elements using already
# declared names (overwriting theme). If it is the case, a warning message is raised.

def restart_default():
    global _default_properties
    _default_properties = dict(
        build_graph=True,
        graph=tf_graph.get_empty_graph(),
        builder_vars=dict(),
        builder_params=dict(),
        is_default=True
    )


# call to the function to restart the default context
restart_default()


# initially, _properties are the default properties
_properties = _default_properties


def is_building_graph():
    return _properties['build_graph']


def is_default():
    return _properties['is_default']


def register_variable(rv):
    # rv is a Random Variable from inferpy
    # the same name cannot be used in builder_vars and builder_params because names are used directly as graph nodes
    if rv.name in _properties['builder_vars'] or rv.name in _properties['builder_params']:
        if is_default() and rv.name not in _properties['builder_params']:
            # in default context; delete variable from builder_vars and graph to add the new one after removal
            del _properties['builder_vars'][rv.name]
            # if update_graph was called and rv name was included in graph, remove it too
            if rv.name in _properties['graph']:
                _properties['graph'].remove_node(rv.name)
            warnings.warn("The variable {} was already defined in the default random variable registry, \
                and is going to be removed. ".format(rv.name))
        else:
            raise ValueError('Random Variable names must be unique among Random Variables and Parameters. \
                             Detected twice: {}'.format(rv.name))
    _properties['builder_vars'][rv.name] = rv


def register_parameter(p):
    # p is a Parameter from inferpy
    # the same name cannot be used in builder_vars and builder_params because names are used directly as graph nodes
    if p.name in _properties['builder_params'] or p.name in _properties['builder_vars']:
        if is_default() and p.name not in _properties['builder_vars']:
            # in default context; delete parameter from builder_params and graph to add the new one after removal
            del _properties['builder_params'][p.name]
            # if update_graph was called and parameter name was included in graph, remove it too
            if p.name in _properties['graph']:
                _properties['graph'].remove_node(p.name)
            warnings.warn("The parameter {} was already defined in the default random parameter registry, \
                and is going to be removed. ".format(p.name))
        else:
            raise ValueError('Parameter names must be unique among Parameters and Random Variables. \
                             Detected twice: {}'.format(p.name))
    _properties['builder_params'][p.name] = p


def get_variable(name):
    return _properties['builder_vars'].get(name, None)


def get_variable_or_parameter(name):
    # return the variable or parameter if exists. Otherwise, return None
    return _properties['builder_vars'].get(
        name,
        _properties['builder_params'].get(name, None)
        )


def get_var_parameters():
    # return a copy of the internal dict properties field 'builder_params', just to
    # avoid the modification of the _properties dict from outside
    return {k: p for k, p in _properties['builder_params'].items()}


def get_graph():
    # return the graph of dependencies of the prob model that is being built
    return _properties['graph']


def update_graph(rv_name=None):
    # update the graph by creating a new one using the actual tf computational graph
    # it uses the actual random variables and parameters, and the rv_name if exists
    # only updates the model if the property build_graph is True
    if _properties['build_graph']:
        # compute all the desired names in the graph (only inferpy RandomVariable and Parameters, and ed2.RandomVariable)
        elements_set = set(_properties['builder_vars']).union(
                           set(_properties['builder_params'])
                          )
        # if rv_name, use this name too
        if rv_name:
            elements_set.add(rv_name)
        # now create the dependencies graph
        _properties['graph'] = tf_graph.get_graph(elements_set)


@contextmanager
def init(graph=None):
    global _properties
    # random variable and parameters registry context. Allows to get access to RVs and parameters as they are built
    # (at the same time ed.tape registers vars)
    # We only allow to use one context level, checked by is_default field (if false, init was called and not exit before)
    assert _properties['is_default']

    # create a new dict, so the default dict is not modified
    _properties = dict()
    _properties['is_default'] = False

    # if graph is not None, use this element as the graph, and do not update it in this context
    _properties['build_graph'] = graph is None
    # if graph is none, start from an empty graph
    if _properties['build_graph']:
        _properties['graph'] = tf_graph.get_empty_graph()
    else:
        _properties['graph'] = graph
    # random variables and parameter dict registry are initially empty
    _properties['builder_vars'] = dict()
    _properties['builder_params'] = dict()
    try:
        yield
    finally:
        # reasign the default registry
        _properties = _default_properties
