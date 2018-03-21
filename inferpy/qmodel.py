import inferpy as inf
import numpy as np
import tensorflow as tf
import edward as ed




class Qmodel(object):

    @staticmethod
    def generate_ed_qvar(v):
        qparams = {}
        for p_name in v.PARAMS:
            if p_name not in ["logits"]:
                qparams.update({p_name: tf.Variable(tf.ones(np.shape(getattr(v, p_name))), dtype="float32")})
        return getattr(ed.models, type(v).__name__)(name = "q_"+str.replace(v.name, ":", ""), **qparams)







