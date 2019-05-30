"""
Any inference class must implement a run method, which receives a sample_dict object,
and returns a dict of posterior objects (random distributions, list of samples, etc.)
"""

from .variational.vi import VI
from .variational.svi import SVI

__all__ = [
    'SVI',
    'VI'
]
