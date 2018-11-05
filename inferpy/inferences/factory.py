import inferpy as inf

CLASS_NAME = "class_name"
PARAMS = "params"
BASE_CLASS_NAME = "base_class_name"
PROPS="properties"





def __add_constructor(cls, class_name, base_class_name, params):

    def constructor(self,*args, **kwargs):


        ## set P and Q


        if Q == None and P != None:
            self.Q = inf.Qmodel.build_from_pmodel(P)
        elif Q != None and P == None:
            self.Q = Q
        else:
            raise ValueError("P or Q must be defined, but not both")

        self.Q = Q




    constructor.__doc__ = "constructor for "+class_name
    constructor.__name__ = "__init__"
    setattr(cls, constructor.__name__, constructor)
