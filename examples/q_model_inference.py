import edward as ed
import inferpy as inf


#### learning a 2 parameters of 1-dim from 2-dim data

N = 50
sampling_mean = [30., 10.]
sess = ed.util.get_session()


with inf.ProbModel() as m:

    theta1 = inf.models.Normal(loc=0., scale=1., dim=1)
    theta2 = inf.models.Normal(loc=0., scale=1., dim=1)


    with inf.replicate(size=N):
        x = inf.models.Normal(loc=[theta1, theta2], scale=1., observed=True)


# define the Qmodel



# define with any type
q_theta1 = inf.Qmodel.Uniform(theta1)
q_theta2 = inf.Qmodel.new_qvar(theta2)


qmodel = inf.Qmodel([q_theta1, q_theta2])

inf.Qmodel.Normal(theta1)

m.compile(Q=qmodel)


x_train = inf.models.Normal(loc=sampling_mean, scale=1.).sample(N)
data = {x.name : x_train}


m.fit(data)

m.posterior(theta1).base_object

m.posterior(theta2)




