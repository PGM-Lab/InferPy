# -*- coding: utf-8 -*-
#

"""
Package with modules defining functions, classes and variables which are
useful for the main functionality provided by inferpy
"""


from inferpy.util.runtime import tf_run_allowed, tf_run_ignored
from . import iterables
from . import random_variable

__all__ = [
    'iterables',
    'random_variable',
    'tf_run_default',
    'tf_run_allowed',
    'tf_run_ignored'
]
