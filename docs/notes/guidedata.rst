Guide to Data Handling
======================

.. code:: python

    import numpy as np
    import inferpy as inf
    from inferpy.models import Normal, InverseGamma, Dirichlet

    #We first define the probabilistic model 
    with inf.ProbModel() as mixture_model:
        # K defines the number of components. 
        K=10
        #Prior for the means of the Gaussians 
        mu = Normal(loc = 0, scale = 1, shape=[K,d])
        #Prior for the precision of the Gaussians 
        invgamma = InverseGamma(concentration = 1, rate = 1, shape=[K,d])
        #Prior for the mixing proportions
        theta = Dirichlet(np.ones(K))

        # Number of observations
        N = 1000
        #data Model
        with inf.replicate(size = N, batch_size = 100)
            # Sample the component indicator of the mixture. This is a latent variable that can not be observed
            z_n = Multinomial(probs = theta)
            # Sample the observed value from the Gaussian of the selected component.  
            x_n = Normal(loc = tf.gather(mu,z_n), scale = tf.gather(invgamma,z_n), observed = true)

    #compile the probabilistic model
    mixture_model.compile(infAlg = 'klqp')

    #fit the model with data
    mixture_model.fit(data)
