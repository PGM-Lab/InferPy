import inferpy as inf
import tensorflow as tf
import edward as ed
import sys




class Qmodel(object):

    def __init__(self, varlist):

        self.varlist = varlist


    @staticmethod
    def build_from_pmodel(p, empirical=False):
        return inf.Qmodel([inf.Qmodel.new_qvar(v) if not empirical else inf.Qmodel.new_qvar_empirical(v,1000) for v in p.latent_vars])



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
    def __generate_ed_qvar(v, initializer, vartype, params):
        qparams = {}

        if initializer == "ones":
            init_f = tf.ones
        elif initializer == "zeroes":
            init_f = tf.zeros
        else:
            raise ValueError("Unsupported initializer: "+initializer)



        for p_name in params:
            if p_name not in ["probs"]:

                p_shape = getattr(v, p_name).shape.as_list() if hasattr(v, p_name) else v.shape


                var = tf.Variable(init_f(p_shape), dtype="float32", name="q_"+v.name+p_name)
                inf.util.Runtime.tf_sess.run(tf.variables_initializer([var]))


                var = tf.where(tf.is_nan(var), tf.zeros_like(var), var)

                if p_name in ["scale"]:
                    var = tf.nn.softplus(var)
                elif p_name in ["probs"]:
                    var = tf.clip_by_value(tf.where(tf.is_nan(var), tf.zeros_like(var), var),1e-8, 1)
                elif p_name in ["logits"]:
                    var = tf.where(tf.is_nan(var), tf.zeros_like(var), var)

                qparams.update({p_name: var})


        qvar =  getattr(ed.models, vartype)(name = "q_"+str.replace(v.name, ":", ""),allow_nan_stats=False, **qparams)

        return qvar




    # rename new_varmodule and new_vartype
    # name


    @staticmethod
    def new_qvar(v, initializer='ones', qvar_inf_module=None, qvar_inf_type = None, qvar_ed_type = None, check_observed = True, name="qvar"):

        if not inf.ProbModel.compatible_var(v):
            raise ValueError("Non-compatible variable")
        elif v.observed and check_observed:
            raise ValueError("Variable "+v.name+" cannot be observed")

        ## default values ##

        if qvar_inf_module == None:
            qvar_inf_module = sys.modules[v.__class__.__module__]

        if qvar_inf_type == None:
            qvar_inf_type = type(v).__name__

        if qvar_ed_type == None:
            qvar_ed_type = type(v).__name__



        qv = getattr(qvar_inf_module, qvar_inf_type)()
        qv.dist = Qmodel.__generate_ed_qvar(v.dist, initializer, qvar_ed_type, v.PARAMS)
        qv.bind = v
        return qv




    @staticmethod
    def new_qvar_empirical(v, n_post_samples, initializer='ones', check_observed = True, name="qvar"):


        if not inf.ProbModel.compatible_var(v):
            raise ValueError("Non-compatible variable")
        elif v.observed and check_observed:
            raise ValueError("Variable "+v.name+" cannot be observed")




        qv = inf.models.Deterministic()


        if initializer == "ones":
            init_f = tf.ones
        elif initializer == "zeroes":
            init_f = tf.zeros
        else:
            raise ValueError("Unsupported initializer: "+initializer)


        dtype = v.base_object.dtype.name
        var = tf.Variable(init_f([n_post_samples] + v.shape, dtype=dtype), dtype=dtype, name="q_"+v.name + "/params")
        qv.base_object = ed.models.Empirical(params=var, name = "q_"+str.replace(v.name, ":", ""))


        qv.bind = v
        return qv



    @staticmethod
    def Empirical(v, n_post_samples=500, initializer='ones'):
        return inf.Qmodel.new_qvar_empirical(v, n_post_samples, initializer)


####





def __add__new_qvar(vartype):

    name = vartype.__name__

    def f(cls,v, initializer='ones'):
        return cls.new_qvar(v, initializer, qvar_inf_type=name)


    f.__doc__ = "documentation for " + name
    f.__name__ = name

    setattr(Qmodel, f.__name__, classmethod(f))


for vartype in inf.models.RandomVariable.__subclasses__():
    __add__new_qvar(vartype)


