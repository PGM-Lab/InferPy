# -*- coding: utf-8 -*-
#


from setuptools import setup

import os
import re
import sys


if sys.version_info < (3, 4):
    sys.exit('Python < 3.4 is not supported')


# get abs path from this folder name
here = os.path.dirname(os.path.abspath(__file__))

# open __init__.py, where version is specified
with open(os.path.join(here, 'inferpy', '__init__.py')) as f:
    txt = f.read()

# try to read it from source code
try:
    version = re.findall(r"^__version__ = '([^']+)'\r?$",
                         txt, re.M)[0]
except IndexError:
    raise RuntimeError('Unable to determine version.')

# get long description from file in docs folder
with open(os.path.join(here, 'docs/project_description.md')) as f:
    long_description = f.read()


# function to read requirements, and include them as package dependencies
def get_requirements(file):
    # read requirements.txt file and return them as a list of strings
    with open(file) as f:
        req = f.readlines()
    return [r.strip() for r in req]  # clean lines from blank spaces and line breaks


setup(
    name='inferpy',
    version=version,
    description='Probabilistic modeling with Tensorflow made easy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Andrés R. Masegosa, Rafael Cabañas, Javier Cózar',
    author_email="andresma@ual.es, rcabanas@ual.es, jcozar87@ual.es",
    url='http://inferpy.readthedocs.io',
    download_url='https://github.com/PGMLabSpain/InferPy/archive/{}.tar.gz'.format(version),
    keywords='machine learning statistics probabilistic programming tensorflow edward2',
    license='Apache License 2.0',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.4'],
    packages=['inferpy'],
    python_requires='>=3.5',
    install_requires=get_requirements('requirements/prod.txt'),
    extras_require={
        'gpu': get_requirements('requirements/gpu.txt'),
        'visualization': get_requirements('requirements/visualization.txt')
    },
    tests_require=get_requirements('requirements/test.txt'),
    include_package_data=True,
)
