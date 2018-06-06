import inferpy as inf
import tensorflow as tf
import edward as ed





class Qmodel(object):

    def __init__(self, varlist):

        self.varlist = varlist


    @staticmethod
    def build_from_pmodel(p):
        return inf.Qmodel([inf.Qmodel.new_qvar(v) for v in p.latent_vars])



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
    def __generate_ed_qvar(v, initializer, vartype):
        qparams = {}

        if initializer == "ones":
            init_f = tf.ones
        elif initializer == "zeroes":
            init_f = tf.zeros
        else:
            raise ValueError("Unsupported initializer: "+initializer)



        #make of type vartype


        for p_name in getattr(inf.models, vartype).PARAMS:
            if p_name not in ["logits"]:

                var = tf.Variable(init_f(v.shape), dtype="float32", name=v.name+"/"+p_name)
                inf.util.Runtime.tf_sess.run(tf.variables_initializer([var]))


                #var = tf.get_variable(v.name+"/"+p_name, v.shape)

                if p_name == "scale":
                    var = tf.nn.softplus(var)

                qparams.update({p_name: var})


        qvar =  getattr(ed.models, vartype)(name = "q_"+str.replace(v.name, ":", ""), **qparams)

        return qvar


 #   @staticmethod
 #   def new_qvar(v, initializer='ones'):
 #       return Qmodel.__new_qvar(v,initializer,None)

    @staticmethod
    def new_qvar(v, initializer='ones', vartype = None, check_observed = True, name="qvar"):

        if not inf.ProbModel.compatible_var(v):
            raise ValueError("Non-compatible variable")
        elif v.observed and check_observed:
            raise ValueError("Variable "+v.name+" cannot be observed")

        if vartype == None:
            vartype = type(v).__name__


        qv = getattr(inf.models, vartype)()
        qv.dist = Qmodel.__generate_ed_qvar(v, initializer, vartype)
        qv.bind = v
        return qv



####





def __add__new_qvar(vartype):

    name = vartype.__name__

    def f(cls,v, initializer='ones'):
        return cls.new_qvar(v,initializer,name)


    f.__doc__ = "documentation for " + name
    f.__name__ = name

    setattr(Qmodel, f.__name__, classmethod(f))


for vartype in inf.models.RandomVariable.__subclasses__():
    __add__new_qvar(vartype)