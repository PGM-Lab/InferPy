import numpy as np
import tensorflow as tf
import inferpy as inf

from abc import abstractproperty, abstractmethod
import edward as ed
import collections



class Param(object):

    def __init__(self, name, value, is_simple=True, duplicate=None):
        self.name = name
        self.is_simple = True if is_simple in [None, True] else False

        elem_ndim = 0 if self.is_simple else 1


        self.p_value = new_ParamValue(value, elem_ndim)

        if duplicate != None:
            self.p_value = self.p_value.duplicate(duplicate)


        self.__input_value = value



    def __repr__(self):

        simple_str = "simple " if self.is_simple else "complex"

        return "<"+simple_str+" param "+self.name+"="+self.p_value.__repr__()+\
               ", ndim="+str(self.ndim)+", td="+str(self.total_dim)+", dim="+str(self.dim)+", bt="+str(self.batches)+">"

    @property
    def nd_range(self):
        if self.is_simple:
            return [0,1]
        return [1,2]

    @property
    def dim_elem(self):
        if self.is_simple:
            n =  1
        elif self.ndim==1:
            n = self.total_dim
        elif isinstance(self.p_value.value, tf.Tensor):
            n = self.p_value.value.get_shape().as_list()[-1]
        elif isinstance(self.p_value.value, collections.Iterable):
            n = self.p_value.value[0].total_dim
        else:
            n = 1


        return n


    @property
    def ndim(self):
        return self.p_value.ndim


    @property
    def total_dim(self):
        return self.p_value.total_dim

    @property
    def dim(self):
        return int(self.total_dim / self.dim_elem)

    @property
    def batches(self):
        return int(self.p_value.batches)


    @property
    def input_value(self):
        return self.__input_value





def new_ParamValue(value, elem_ndim=0):

    if type(value) in [type([]), type(np.array([]))]:
        p = ParamValueArray(value, elem_ndim)
    elif np.isscalar(value):
        p = ParamValueScalar(value)
    elif isinstance(value, tf.Tensor):
        p = ParamValueTensor(value)
    elif isinstance(value, inf.models.RandomVariable):
        p = ParamValueInfVar(value)
    elif isinstance(value, ed.models.RandomVariable):
        p = ParamValueTensor(value)

    else:
        raise ValueError("Wrong parameter value")

    return p

class ParamValue(object):

    @abstractproperty
    def ndim(self):
        pass

    @abstractproperty
    def total_dim(self):
        pass


    @abstractproperty
    def batches(self):
        pass

    def __repeat(self, k):
        v = [p for p in np.repeat(self, k)]
        ret = ParamValueArray([], elem_ndim=None)
        ret.value = v
        return ret

    def repeat(self, kd=1, kb=1):
        ret = self

        if kb == 1 and kd == 1:
            return ret




        if kd > 1 or kb > 1 and self.ndim==0: ret = ret.__repeat(kd)
        if kb > 1: ret = ret.__repeat(kb)

        return ret


    def __repr__(self):
        return str(self.value)


    def get_std_value(self):
        return self.value

    def get_param_tensor(self):
        return self.tensor







class ParamValueScalar(ParamValue):

    def __init__(self,value):
        self.value = value

    @abstractproperty
    def ndim(self):
        return 0


    @abstractproperty
    def total_dim(self):
        return 1


    @abstractproperty
    def batches(self):
        return 1

    def all_scalar(self):
        return True

    @property
    def tensor(self):
        return tf.constant([self.value], dtype="float32")

    def get_param_tensor(self):
        return tf.constant([self.value], dtype="float32")

    def duplicate(self, op="equal"):

        if op=="equal":
            new_value = self.value
        elif op=="negative":
            new_value = -self.value
        elif op == "prob_complement":
            new_value = 1-self.value


        return new_ParamValue([self.value, new_value], 0)


class ParamValueArray(ParamValue):

    def __init__(self, value, elem_ndim=0):
        self.value = []
        for v in value:
            child_p = new_ParamValue(v)

        #    if child_p.ndim != elem_ndim and elem_ndim != None:
        #        raise ValueError("Array Parameter with wrong number of dimension")

            self.value.append(child_p)

    @abstractproperty
    def ndim(self):
        return np.max([v.ndim+1 for v in self.value])

    @abstractproperty
    def total_dim(self):
        return np.sum([v.total_dim for v in self.value])


    @abstractproperty
    def batches(self):
        return self.value[0].total_dim


    def all_scalar(self):
        return np.all([v.all_scalar() for v in self.value])

    def get_std_value(self):
        return [v.get_std_value() for v in self.value]


    @property
    def tf_array(self):
        return np.array([v.tf_array if isinstance(v, ParamValueArray) else v.tensor for v in self.value])

    @property
    def tensor(self):
        if self.all_scalar():
            v = np.array(self.get_std_value())
            return tf.constant(v, dtype="float32")
        else:

            tf_array = self.tf_array
            shape = tf_array.shape
            tf_vect = list(tf_array.flatten())

            if shape == (1,):
                return tf_vect[0]


            m = np.prod([x for x in shape])
            if  m != self.total_dim:
                shape = tuple([x for x in shape] + [self.total_dim / m])


            return tf.reshape(tf.stack(tf_vect), shape)




class ParamValueTensor(ParamValue):

    def __init__(self, value):
        if value.shape == ():
            value = tf.reshape(value, shape=(1,))

        self.value = value

    @abstractproperty
    def ndim(self):
        shape = self.value.shape.as_list()
        return 0 if shape in ([], [1])  else  len(shape)


    @abstractproperty
    def total_dim(self):
        return 1 if len(self.value.shape.as_list()) == 0 else self.value.shape.as_list()[-1]

    @abstractproperty
    def batches(self):
        return 1 if len(self.value.shape.as_list()) < 2 else self.value.shape.as_list()[0]

    def all_scalar(self):
        return False

    @property
    def tensor(self):
        return self.value

    def get_param_tensor(self):

        v = self.value

        if self.value.shape == ():
            v = tf.reshape(v, shape=(1,1))

        return v

    def duplicate(self, op="equal"):

        if op == "equal":
            new_value = self.value
        elif op == "negative":
            new_value = -self.value
        elif op == "prob_complement":
            new_value = 1 - self.value

        return new_ParamValue([self.value, new_value], 0)


class ParamValueInfVar(ParamValue):

    def __init__(self, value):
        self.value = value

    @abstractproperty
    def ndim(self):
        shape = self.value.shape
        return 0 if shape in ([], [1])  else  len(shape)

    @abstractproperty
    def total_dim(self):
        return self.value.dim

    @abstractproperty
    def batches(self):
        return self.value.batches

    def all_scalar(self):
        return False

    @property
    def tensor(self):
        return self.value.base_object



class ParamValueEdVar(ParamValue):

    def __init__(self, value):
        if value.shape == ():
            value = tf.reshape(value, shape=(1,))


    @abstractproperty
    def ndim(self):
        shape = self.value.shape
        return 0 if shape in ([], [1])  else  len(shape)

    @abstractproperty
    def total_dim(self):
        return 1 if len(self.value.shape.as_list()) == 0 else self.value.shape.as_list()[-1]

    @abstractproperty
    def batches(self):
        return 1 if len(self.value.shape.as_list()) < 2 else self.value.shape.as_list()[0]

    def all_scalar(self):
        return False

    @property
    def tensor(self):
        return self.value

    def get_param_tensor(self):

        v = self.value

        if self.value.shape == ():
            v = tf.reshape(v, shape=(1,1))

        return v





class ParamList(object):

    def __init__(self, params, args_list=[], kwargs_dict={}, is_simple={}, param_dim=None):

        plist=[]

        for p_name in params:
            if len(args_list) > 0:
                if p_name in kwargs_dict:
                    raise ValueError("Wrong positional or keyword argument")

                plist.append(Param(name=p_name, value = args_list[0], is_simple=is_simple.get(p_name)))
                args_list = args_list[1:]
            else:
                if p_name in kwargs_dict:
                    plist.append(Param(name=p_name, value = kwargs_dict.get(p_name), is_simple=is_simple.get(p_name)))


        self.plist=plist
        self.param_dim = param_dim


    def __repr__(self):
        str = ""
        for p in self.plist:
            str =str+" "+p.__repr__()+",\n"

        return "["+str[1:-2]+"]"


    def is_empty(self):
        return len(self.plist)==0


    def check_params(self):
        self.__check_ndim()
        self.__check_batches()
        self.__check_dim()


    def __check_ndim(self):
        for p in self.plist:

            n = p.ndim
            low = p.nd_range[0]
            up = p.nd_range[1]

            if isinstance(p.p_value, ParamValueInfVar) and p.batches>1 and p.batches==inf.replicate.get_total_size():
                low = low+1
                up = up+1

            if n < low or n > up:
                raise ValueError(
                    "Wrong dimension of "+p.name+" which is " + str(n) + " but should be in interval [" + str(low) + "," + str(
                        up) + "]")


    def __check_batches(self):

        N = self.final_shape[0]
        param_batches = np.unique([p.batches for p in self.plist])



        if len([x for x in param_batches if x not in [1,N]]) > 0:
            raise ValueError("Error: the number of batches for all the parameters must be 1 or "+str(N))



    def __check_dim(self):
        D = self.final_shape[1]
        param_dim = np.unique([p.dim for p in self.plist])

        if len([x for x in param_dim if x not in [1,D]]) > 0:
            raise ValueError("Inconsistent parameter dimensions")


    @property
    def final_shape(self):
        N = inf.replicate.get_total_size()
        D = np.max([v.dim for v in self.plist])

        if self.param_dim != None:
            D = np.max([D, self.param_dim])

        return [N,D]


    def get_reshaped_param_dict(self):
        d = {}
        final_shape = self.final_shape

        D = final_shape[1]
        N = final_shape[0]

        for p in self.plist:

            v = p.p_value.repeat(D / p.dim, N / p.batches).get_param_tensor()


            # complex params should have at least 2 dimensions
            if not p.is_simple and len(v.shape.as_list())<2:
                v = tf.stack([v])



            d.update({p.name : v})

        return d



