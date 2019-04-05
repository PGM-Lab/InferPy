from contextlib import contextmanager


@contextmanager
def no_raised_exc():
    yield
