import inferpy as inf
import numpy as np
import tensorflow as tf
import edward as ed

import inferpy.inferences




class Qmodel(object):

    def __init__(self, varlist):

        self.varlist = varlist


    @staticmethod
    def build_from_pmodel(p):
        return inferpy.inferences.Qmodel([inferpy.inferences.Qmodel.new_qvar(v) for v in p.latent_vars])



    @property
    def dict(self):
        d = {}
        for v in self.varlist:
            d.update({v.bind.dist : v.dist})
        return d




    @property
    def varlist(self):
        """ list of variables"""
        return self.__varlist

    @varlist.setter
    def varlist(self,varlist):
        """ modifies the list of variables in the model"""
        for v in varlist:
            if not Qmodel.compatible_var(v):
                raise ValueError("Non-compatible var "+v.name)

        self.__varlist = varlist


    def add_var(self, v):
        """ Method for adding a new q variable """

        if not Qmodel.compatible_var(v):
            raise ValueError("The input argument must be a q variable")

        if v not in self.varlist:
            self.varlist.append(v)



    @staticmethod
    def compatible_var(v):
        return inf.ProbModel.compatible_var(v) and v.bind != None



    @staticmethod
    def __generate_ed_qvar(v, initializer):
        qparams = {}

        if initializer == "ones":
            init_f = tf.ones
        elif initializer == "zeroes":
            init_f = tf.zeros
        else:
            raise ValueError("Unsupported initializer: "+initializer)




        for p_name in v.PARAMS:
            if p_name not in ["logits"]:
                qparams.update({p_name: tf.Variable(init_f(np.shape(getattr(v, p_name))), dtype="float32")})

        qvar =  getattr(ed.models, type(v).__name__)(name = "q_"+str.replace(v.name, ":", ""), **qparams)

        return qvar


    @staticmethod
    def new_qvar(v, initializer='ones'):

        if not inf.ProbModel.compatible_var(v):
            raise ValueError("Non-compatible variable")
        elif v.observed:
            raise ValueError("Variable "+v.name+" cannot be observed")


        qv = getattr(inf.models, type(v).__name__)()
        qv.dist = Qmodel.__generate_ed_qvar(v, initializer)
        qv.bind = v
        return qv













