Guiding Principles
==================


Features
~~~~~~~~~~~~

The main features of InferPy are listed below.

-  The models that can be defined in Inferpy are those that can be defined using Edward, whose probability distribuions
   are mainly inherited from TensorFlow Distribuitons package.
-  Edward's drawback is that for the model definition, the user has to manage complex multidimensional arrays called
   tensors. By contrast, in InferPy all the parameters in a model can be defined using the standard Python types
   (compatibility with Numpy is available as well).
-  InferPy directly relies on top of Edward's inference engine and
   includes all the inference algorithms included in this package. As
   Edward's inference engine relies on TensorFlow computing engine,
   InferPy also relies on it too.
-  InferPy seamlessly process data contained in a numpy array, Tensorflow's
   tensor, Tensorflow's Dataset (tf.Data API), Pandas' DataFrame or Apache Spark's
   DataFrame.
-  InferPy also includes novel distributed statistical inference
   algorithms by combining Tensorflow computing
   engines.



Architecture
~~~~~~~~~~~~~~~

Given the previous considerations, we might summarize the InferPy architecture as follows.



.. figure:: ../_static/img/inferpy_architecture_simple.png
   :alt: InferPy architecture
   :scale: 35 %
   :align: center


Note that InferPy can be seen as an upper layer for working with probabilistic distributions defined
over tensors. Most of the interaction is done with Edward:  the definitions of the distributions, the
inference. However, InferPy also interacts directly with Tensorflow in some operations that are hidden to
the user, e.g. the manipulation of the tensors representing the parameters of the distributions.

An additional advantage of using Edward and Tensorflow as inference engine, is that all the paralelisation details
are hidden to the user. Moreover, the same code will run either in CPUs or GPUs.


For some less important task, InferPy might also interact with other third-party software. For example, reading data is
done with Pandas or the visualization tasks are leveraged to MatPlotLib.

