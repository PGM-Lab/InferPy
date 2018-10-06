# -*- coding: utf-8 -*-
#

from inferpy.models.random_variable import *
#from inferpy.models.normal import *
from inferpy.models.deterministic import *
from inferpy.models.factory import *


from inferpy.models import predefined


# IMPLEMENTED RANDOM VARIABLES



ALLOWED_VARS = sorted(list(np.unique([cls.__name__ for cls in RandomVariable.__subclasses__()])))




