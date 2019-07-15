Guide to Data Handling
======================

The module ``inferpy.data.loaders`` provides the functionality for handling data. In particular,
all the classes for loading data will inherit from the class ``DataLoader`` defined at
this module.


CSV files
---------------

Data can be loaded from CSV files by means of the class ``CsvLoader`` whose
object can be built as follows:



.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 6-8


where ``path`` can be a string indicating the location of the csv file or
also a list of strings (i.e., for datasets distributed across multiple CSV files):



.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 10-11



A data loader can be built indistinctly from  CSV files with or without header.
However, in case of a list of files,  the presence of the header and column names must
be consistent among all the files.


When loading data from a CSV file, it could happen that we need to
map the columns in the dataset to another set of variables. This can be made
by means of the input argument ``var_dict``, which is a dictionary where the
keys are the variable names and the values are lists of integer indicating
the columns (starting by 0). For example, in a data set whose columns names
are ``"x"`` and ``"y"``, we might be interested in rename them:



.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 44


This mapping functionality can be used for grouping columns into a single
variable:

.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 51






Data in memory
------------------

Analogously, a data loader can be built from data already loaded into memory, e.g.,
pandas data. For this, we will use the class ``SampleDictLoader`` which can be
instantiated as follows.


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 16-19

Properties
---------------

From any object of class ``DataLoader`` we might obain the size, (i.e., number of instances)
of the list of variable names:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 33-36


In case of a ``CsvLoader``, we might determine if the source files have or not
a header:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 37-38



Extracting data
-------------------

Data can be loader as a dictionary (of numpy objects) or as tensorflow dataset object:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 66-72


Usage with probabilistic models
----------------------------------

Clearly, the final goal of loading data is learning a probabilistic model.
Thus, consider the following code of a simple linear regression:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 83-106


We have seen so far how to make inference by invoking the method ``fit`` which
takes a dictionary of samples as input parameter:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 109


The ``data`` parameter can be straightforward replaced by an object of
class ``DataLoader``:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 113-114


Note that the column names must be the same than those in our model. In case
of being different or reading from a file without header, we might use
the mapping functionality:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 118-119








