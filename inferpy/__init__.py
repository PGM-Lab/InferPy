# -*- coding: utf-8 -*-
#


__version__ = '0.2.2'
VERSION = __version__


from inferpy import criticism
from inferpy import models
from inferpy import util
from inferpy import inferences

#from inferpy.inferences import inference


# Direct imports for convenience
from inferpy.criticism.evaluate import *
from inferpy.util.runtime import get_session
from inferpy.models.prob_model import *
from inferpy.models.replicate import *

from inferpy.inferences.inference import INF_METHODS
from inferpy.inferences.inference import INF_METHODS_ALIAS
from inferpy.inferences.qmodel import *



## basic funcitons

from inferpy.util.ops import *




