from contextlib import contextmanager


# This dict store the active (if active is True) context parameters of prob model builder,
# the model built as the initial model, and the builder vars that are being built now (model expanded?)
_properties = dict(
    active=False,
    model=None,
    builder_vars=None
)


def is_active():
    # is the prob model builder context active?
    return _properties['active']


def register_variable(rv):
    # TODO: if not active, raise custom exception
    # rv is a Random Variable from inferpy
    _properties['builder_vars'][rv.name] = rv


def get_builder_variable(name):
    # TODO: if not active, raise custom exception
    # return the variable if exists. Otherwise, return None
    return _properties['builder_vars'].get(name, None)


def get_graph():
    # TODO: if not active, raise custom exception
    # return the graph of dependencies of the prob model that is being built
    return _properties['model'].graph


def get_model_variable(name):
    # TODO: if not active, raise custom exception
    # return the graph of dependencies of the prob model that is being built
    return _properties['model'].vars.get(name, None)


@contextmanager
def builder(model):
    # prob model builder context. Allows to get access to RVs as they are built (at the same time ed.tape registers vars)
    # We only allow to use one context level
    assert not _properties['active']
    _properties['active'] = True
    _properties['model'] = model
    _properties['builder_vars'] = dict()
    try:
        yield
    finally:
        _properties['active'] = False
        _properties['model'] = None
        _properties['builder_vars'] = None
