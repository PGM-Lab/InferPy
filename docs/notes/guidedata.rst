Guide to Data Handling
======================

The module ``inferpy.data.loaders`` provides the basic functionality for handling data. In particular,
all the classes for loading data will inherit from the class ``DataLoader`` defined at
this module.


CSV files
---------------

Data can be loaded from CSV files through the class ``CsvLoader`` whose
objects can be built as follows:



.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 6-8


where ``path`` can be either a string indicating the location of the csv file or
a list of strings (i.e., for datasets distributed across multiple CSV files):



.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 10-11



A data loader can be built from  CSV files with or without a header.
However, in case of a list of files,  the presence of the header and column names must
be consistent among all the files.


When loading data from a CSV file, we might need to
map the columns in the dataset to another set of variables. This can be made
using the input argument ``var_dict``, which is a dictionary where the
keys are the variable names and the values are lists of integers indicating
the columns (0 stands for the first data column). For example, in a data set whose columns names
are ``"x"`` and ``"y"``, we might be interested in renaming them:



.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 44


This mapping functionality can also be used for grouping columns into a single
variable:

.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 51






Data in memory
------------------

Analogously, a data loader can be built from data already loaded into memory, e.g.,
pandas data. To do this, we will use the class ``SampleDictLoader`` which can be
instantiated as follows.


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 16-19

Properties
---------------

From any object of class ``DataLoader`` we can obtain the size, (i.e., number of instances)
of the list of variable names:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 33-36


In case of a ``CsvLoader``, we can determine if the source files have or not
a header:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 37-38



Extracting data
-------------------

Data can be loaded as a dictionary (of numpy objects) or as TensorFlow dataset object:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 66-72


Usage with probabilistic models
----------------------------------

Making inference in a probabilistic model is the final goal of loading data.
Consider the following code of a simple linear regression:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 83-106


We have seen so far that, for making inference we invoke the method ``fit`` which
takes a dictionary of samples as an input parameter:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 109


The ``data`` parameter can be replaced by an object of
class ``DataLoader``:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 113-114


Note that column names must be the same as those in the model. In case
of being different or reading from a file without header, we use
the mapping functionality:


.. literalinclude:: ../../examples/docs/guidedata/1.py
   :language: python3
   :lines: 118-119








