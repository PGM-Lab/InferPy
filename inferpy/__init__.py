# -*- coding: utf-8 -*-
#


from inferpy import models
from inferpy import util
#from inferpy.inferences import inference


from inferpy.replicate import *
from inferpy.version import *
from inferpy.prob_model import *
from inferpy.qmodel import *

## basic funcitons

from inferpy.util.ops import *



INF_METHODS = ["KLpq", "KLqp", "Laplace", "ReparameterizationEntropyKLqp", "ReparameterizationKLKLqp", "ReparameterizationKLqp", "ScoreEntropyKLqp", "ScoreKLKLqp", "ScoreKLqp", "ScoreRBKLqp", "WakeSleep"]
