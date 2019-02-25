import inferpy as inf


with inf.ProbModel() as m:
    theta = inf.models.Beta(0.5,0.5)
    z =  inf.models.Categorical(probs=[theta, 1-theta], name="z")


m.sample()


# Categorical variable depending on another categorical variable

with inf.ProbModel() as m2:
    y =  inf.models.Categorical(probs=[0.4,0.6], name="y")
    x = inf.models.Categorical(probs=inf.case({y.equal(0): [0.0, 1.0],
                                               y.equal(1): [1.0, 0.0] }), name="x")
m2.sample()


# Categorical variable depending on a Normal distributed variable

with inf.ProbModel() as m3:
    a = inf.models.Normal(0,1, name="a")
    b = inf.models.Categorical(probs=inf.case({a>0: [0.0, 1.0],
                                               a<=0: [1.0, 0.0]}), name="b")
m3.sample()


# Normal distributed variable depending on a Categorical variable

with inf.ProbModel() as m4:
    d =  inf.models.Categorical(probs=[0.4,0.6], name="d")
    c = inf.models.Normal(loc=inf.case({d.equal(0): 0.,
                                        d.equal(1): 100.}), scale=1., name="c")
m4.sample()




with inf.ProbModel() as m:
    y =  inf.models.Categorical(probs=[0.4,0.6], name="y")
    with inf.replicate(size=10):
        x = inf.models.Categorical(probs=inf.case({y.equal(0): [0.5, 0.5],
                                                    y.equal(1): [1.0, 0.0] }), name="x")
m.sample()



with inf.ProbModel() as m:
    y =  inf.models.Categorical(probs=[0.4,0.6], name="y")
    x = inf.models.Categorical(probs=inf.case_states(y, {0: [0.0, 1.0],
                                                         1: [1.0, 0.0] }), name="x")
m.sample()


with inf.ProbModel() as m:
    y =  inf.models.Categorical(probs=[0.4,0.6], name="y")
    with inf.replicate(size=10):
        x = inf.models.Categorical(probs=inf.gather([[0.5, 0.5], [1.0, 0.0]], y), name="x")

m.sample()





y =  inf.models.Categorical(probs=[0.5,0.5], name="y", dim=2)
p = inf.case_states(y, {(0,0): [1.0, 0.0, 0.0, 0.0], (0,1): [0.0, 1.0, 0.0, 0.0],
                        (1, 0): [0.0, 0.0, 1.0, 0.0], (1,1): [0.0, 0.0, 0.0, 1.0]} )

with inf.replicate(size=10):
    x = inf.models.Categorical(probs=p, name="x")

####

y =  inf.models.Categorical(probs=[0.5,0.5], name="y", dim=1)
z =  inf.models.Categorical(probs=[0.5,0.5], name="z", dim=1)


p = inf.case_states((y,z), {(0,0): [1.0, 0.0, 0.0, 0.0], (0,1): [0.0, 1.0, 0.0, 0.0],
                            (1, 0): [0.0, 0.0, 1.0, 0.0], (1,1): [0.0, 0.0, 0.0, 1.0]} )

with inf.replicate(size=10):
    x = inf.models.Categorical(probs=p, name="x")

####

p = inf.case_states([y,z], {(0,0): [1.0, 0.0, 0.0, 0.0], (0,1): [0.0, 1.0, 0.0, 0.0],
                            (1, 0): [0.0, 0.0, 1.0, 0.0], (1,1): [0.0, 0.0, 0.0, 1.0]} )

with inf.replicate(size=10):
    x = inf.models.Categorical(probs=p, name="x")

