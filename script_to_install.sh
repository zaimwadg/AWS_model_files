#!/bin/bash


wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh

expect install_anaconda

source .bashrc

jupyter notebook --generate-config

cd /home/ubuntu/.jupyter/

echo -e "c = get_config()\nc.IPKernelApp.pylab = 'inline'\nc.NotebookApp.ip = '*'\nc.NotebookApp.open_browser = False\nc.NotebookApp.port = 8888" >> jupyter_notebook_config.py

cd ..

expect python_3

expect install_scikit