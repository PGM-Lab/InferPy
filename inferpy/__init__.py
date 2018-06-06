# -*- coding: utf-8 -*-
#

from inferpy import criticism
from inferpy import models
from inferpy import util

#from inferpy.inferences import inference


# Direct imports for convenience
from inferpy.criticism.evaluate import *
from inferpy.util.runtime import get_session
from inferpy.models.prob_model import *
from inferpy.models.qmodel import *
from inferpy.models.replicate import *


from inferpy.version import *

## basic funcitons

from inferpy.util.ops import *



INF_METHODS = ["KLpq", "KLqp", "Laplace", "ReparameterizationEntropyKLqp", "ReparameterizationKLKLqp", "ReparameterizationKLqp", "ScoreEntropyKLqp", "ScoreKLKLqp", "ScoreKLqp", "ScoreRBKLqp", "WakeSleep"]
