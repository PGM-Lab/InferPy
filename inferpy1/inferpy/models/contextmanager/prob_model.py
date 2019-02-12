from contextlib import contextmanager
from inferpy import exceptions
from inferpy.util import tf_graph


# This dict store the active (if active is True) context parameters of prob model builder,
# the model built as the initial model, and the builder vars that are being built now (model expanded?)
_properties = dict(
    active=False,
    graph=None,
    builder_vars=None
)


def is_active():
    # is the prob model builder context active?
    return _properties['active']


def register_variable(rv):
    # TODO: if not active, raise custom exception
    # rv is a Random Variable from inferpy
    if rv.name in _properties['builder_vars']:
        raise exceptions.NotUniqueRandomVariableName(
            'Random Variable names must be unique. Detected twice: {}'.format(rv.name))
    _properties['builder_vars'][rv.name] = rv


def get_builder_variable(name):
    # TODO: if not active, raise custom exception
    # return the variable if exists. Otherwise, return None
    return _properties['builder_vars'].get(name, None)


def get_graph():
    # TODO: if not active, raise custom exception
    # return the graph of dependencies of the prob model that is being built
    return _properties['graph']


def update_graph(rv_name):
    # TODO: if not active, raise custom exception
    # return the graph of dependencies of the prob model that is being built
    # create a new graph using the actual tf computational graph, and clean it using the actual
    # builder vars and the random var name which is being built

    _properties['graph'] = tf_graph.get_graph(set(_properties['builder_vars']).union(set(rv_name)))


@contextmanager
def builder():
    # prob model builder context. Allows to get access to RVs as they are built (at the same time ed.tape registers vars)
    # We only allow to use one context level
    assert not _properties['active']
    _properties['active'] = True
    _properties['graph'] = tf_graph.get_empty_graph()
    _properties['builder_vars'] = dict()
    try:
        yield
    finally:
        _properties['active'] = False
        _properties['graph'] = None
        _properties['builder_vars'] = None
