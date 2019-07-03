#!/usr/bin/env bash

#py_folder="../python_documentation/"
py_folder="../inferpy/"
rst_folder="./modules/"

rm -f ${rst_folder}* && sphinx-apidoc -f -o ${rst_folder} ${py_folder} && make clean && make html
