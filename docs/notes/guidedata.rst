Guide to Data Handling
======================

InferPy leverages existing Pandas_ functionality for reading data. As a consequence, InferPy can
learn from datasets in any file format handled by Pandas. This is possible because the method
``inferpy.ProbModel.fit(data)`` accepts as input argument a Pandas DataFrame.

.. _Pandas: https://pandas.pydata.org


In the following code fragment, an example of learning a model from a CVS file is shown:

.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3