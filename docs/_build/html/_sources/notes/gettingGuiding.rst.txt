Guiding Principles
==================

-  InferPy's probability distribuions are mainly inherited from
   TensorFlow Distribuitons package. 
   .. InferPy's API is fully compatible with tf.distributions' API. The 'shape' argument was added as a simplifing option when defining multidimensional distributions.
-  InferPy directly relies on top of Edward's inference engine and
   includes all the inference algorithms included in this package. As
   Edward's inference engine relies on TensorFlow computing engine,
   InferPy also relies on it too.
-  InferPy seamsly process data contained in a numpy array, Tensorflow's
   tensor, Tensorflow's Dataset (tf.Data API), Pandas' DataFrame or Apache Spark's
   DataFrame.
-  InferPy also includes novel distributed statistical inference
   algorithms by combining Tensorflow and Apache Spark computing
   engines.