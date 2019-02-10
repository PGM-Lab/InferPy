from inferpy import models
from inferpy.models import contextmanager


def test_active_context():
    # Initially data_model context disabled
    assert not contextmanager.data_model.is_active()

    # inside context, it is True
    with models.datamodel():
        assert contextmanager.data_model.is_active()
    # and now false again
    assert not contextmanager.data_model.is_active()
