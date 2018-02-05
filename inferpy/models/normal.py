from inferpy.replicate import *
import inferpy.config as inf_conf
import edward.models as base_models
import numpy as np

class Normal:
	def __init__(self, loc, scale, dim=None, name="inf_Normal"):

		self.__check_params(loc, scale, dim)

		param_dim = 1
		if dim!=None: param_dim

		# shape = (batches, dimension)
		self_shape = (replicate.getSize(), np.max([np.size(loc), np.size(scale), param_dim]))

		# build the loc and scale matrix
		if np.isscalar(loc):
			loc_rep = np.tile(loc, (self_shape[0], self_shape[1]))
		else:
			loc_rep =  np.tile(loc, (self_shape[0], 1))

		if np.isscalar(scale):
			scale_rep = np.tile(scale, (self_shape[0], self_shape[1]))
		else:
			scale_rep = np.tile(scale, (self_shape[0], 1))

		# build the distribution

		self.__dist = base_models.Normal(loc=loc_rep, scale=scale_rep, name=name)

	# getter methods


	@property
	def loc(self):
		return inf_conf.tf_sess.run(self.dist.loc)

	@property
	def scale(self):
		return inf_conf.tf_sess.run(self.dist.scale)

	@property
	def dim(self):
		return self.dist.shape.as_list()[1]
	@property
	def batches(self):
		return self.dist.shape.as_list()[0]

	@property
	def shape(self):
		#return self.__shape
		return self.dist.shape.as_list()

	@property
	def dist(self):
		return self.__dist


	def __check_params(self, loc, scale, dim):

		# loc and scale cannot be multidimensional arrays (by now)
		if np.ndim(loc) > 1 or np.ndim(scale) > 1:
			raise ValueError("loc and scale cannot be multidimensional arrays")

		len_loc = np.size(loc)
		len_scale = np.size(scale)

		# loc and scale lengths must be equal or must be scalars
		if len_loc > 1 and len_scale > 1 and len_loc != len_scale:
			raise ValueError("loc and scale lengths must be equal or must be 1")

		# loc can be a scalar or a vector of length dim

		if dim != None and len_loc > 1 and dim != len_loc:
			raise ValueError("loc length is not consistent with value in dim")

		if dim != None and len_scale > 1 and dim != len_scale:
			raise ValueError("scale length is not consistent with value in dim")


	def sample(self, v):
		return inf_conf.tf_sess.run(self.dist.sample(v))


