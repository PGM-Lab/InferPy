# -*- coding: utf-8 -*-
#

"""
Package with modules defining functions, classes and variables which are
useful for the main functionality provided by inferpy
"""


from inferpy.util.runtime import tf_run_default, tf_run_eval
from inferpy.util.wrappers import tf_run_wrapper
from . import iterables

__all__ = [
    'iterables',
    'tf_run_default',
    'tf_run_eval',
    'tf_run_wrapper'
]