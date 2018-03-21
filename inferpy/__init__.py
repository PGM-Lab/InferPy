# -*- coding: utf-8 -*-
#


from inferpy import models
from inferpy import util

from inferpy.replicate import *
from inferpy.prob_model import *

from inferpy.version import *

from inferpy.qmodel import *


## basic funcitons

def matmul(A,B):
    return A.__matmul__(B)