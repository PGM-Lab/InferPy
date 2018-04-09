import inferpy as inf

# define a 2-dimension Normal distribution
with inf.replicate(size=2):
	with inf.replicate(size=3):
		x = inf.models.Normal(loc=0., scale=1., dim=2)


# print its parameters
print(x.loc)
print(x.scale)



# the shape of the distribution is (6,2)
print(x.shape)

# get a sample
sample_x = x.sample([4,10])

# the shape of the sample is (4, 10, 6, 2)
print(sample_x.shape)


x.sample(1, tf_run=False)
x.sample(1)

# probability and log probability of the sample
x.prob(sample_x)

x.log_prob(sample_x)


# get the underlying distribution Edward object
ed_x = x.dist



