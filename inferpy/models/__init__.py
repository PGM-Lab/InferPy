# -*- coding: utf-8 -*-
#

from inferpy.models.random_variable import *
#from inferpy.models.normal import *
from inferpy.models.deterministic import *
from inferpy.models.factory import *





# IMPLEMENTED RANDOM VARIABELS



ALLOWED_VARS = list(np.unique([cls.__name__ for cls in RandomVariable.__subclasses__()]))




