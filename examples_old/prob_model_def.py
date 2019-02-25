import inferpy as inf
from inferpy.models import Normal



with inf.ProbModel() as m:

    x = Normal(loc=1., scale=1., name="x", observed=True)
    y = Normal(loc=x, scale=1., dim=3, name="y")


# print the list of variables
print(m.varlist)
print(m.latent_vars)
print(m.observed_vars)

# get a sample

m_sample = m.sample()



# compute the log_prob for each element in the sample
print(m.log_prob(m_sample))

# compute the sum of the log_prob
print(m.sum_log_prob(m_sample))




### alternative definition

x2 = Normal(loc=1., scale=1., name="x2", observed=True)
y2 = Normal(loc=x, scale=1., dim=3, name="y2")

m2 = inf.ProbModel(varlist=[x2,y2])

