# -*- coding: utf-8 -*-
#


from setuptools import setup, find_packages

import os
import re


here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(here, 'inferpy', '__init__.py')) as f:
    txt = f.read()

try:
    version = re.findall(r"^__version__ = '([^']+)'\r?$",
                         txt, re.M)[0]
except IndexError:
    raise RuntimeError('Unable to determine version.')


with open(os.path.join(here, 'inferpy/docs/project_description.md')) as f:
    long_description = f.read()



setup(
    name='inferpy',
    version=version,
    description='Probabilistic modeling with Tensorflow made easy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Andrés R. Masegosa, Rafael Cabañas',
    author_email="andresma@ual.es, rcabanas@ual.es",
    install_requires=['tensorflow >= 1.5, <1.8', 'numpy>=1.14', 'edward==1.3.5', 'pandas>0.15.0'],
    extras_require={
        'tensorflow with gpu': ['tensorflow-gpu  >= 1.5, <1.8'],
        'visualization': ['matplotlib>=1.3',
                          'pillow>=3.4.2',
                          'seaborn>=0.3.1']},
    tests_require=['pytest', 'pytest-pep8'],
    url='http://inferpy.readthedocs.io',
    download_url='https://github.com/PGMLabSpain/InferPy/archive/' + version + '.tar.gz',
    keywords='machine learning statistics probabilistic programming tensorflow edward',
    license='Apache License 2.0',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
    packages=find_packages(exclude=['playground_ignored', ]),
    package_data={'inferpy': ['docs/*.md']},
)






