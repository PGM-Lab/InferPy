from contextlib import contextmanager


"""This context must be used by the prob model before calling the builder function.
This is used to store the layers.sequential.Sequential objects...
"""


def _restart_properties():
    global _properties
    _properties = dict(
        _sequentials=[],
        enabled=False
    )


# call to the function to restart the default context
_restart_properties()


def add_sequential(sequential):
    # only if enabled append sequential object (i.e. when building the graph of dependencies is not necessary)
    if _properties["enabled"]:
        _properties["_sequentials"].append(sequential)


def get_losses():
    assert _properties["enabled"]
    losses = [loss for sequential in _properties["_sequentials"] for loss in sequential.losses]
    return sum(losses) if len(losses) > 0 else None


@contextmanager
def init(graph=None):
    global _properties

    assert not _properties["enabled"]
    assert _properties["_sequentials"] == []

    try:
        # now the sequentials created can use this list to store the objects
        _properties["enabled"] = True
        yield
    finally:
        # reasign the default object
        _restart_properties()
