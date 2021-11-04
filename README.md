CODE FOR QOVAE

## System Requirements

code was ran on linux operating system using ubuntu:18.04 but should work on windows 10 as well.

## Installation Guide

You can use the environment.yml to create a conda env for running the code using

conda env create -f environment.yml

(should take less than an hour to install)

## Demo and Running the code

Once environment is set up run train_qo.py in terminal and the code will

1) load the entire entangled dataset and train on it
2) once finished training the code will sample 1000 experiments from the model and  
3) calculate their entanglement and display some metrics

Should take no longer than one day to train and calculate (depending on model size and using a GPU)
