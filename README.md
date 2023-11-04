# TensorFluid
Implementation of lattice-boltzmann method in python, Tensorflow and C. 

# Installation 
Run env-install.sh

# Timings summary 
Timings for flow around bluff body, problem size 1680 x 720 

## CPU Tensorflow
with export OMP_NUM_THREADS=1
134s

## GPU Tensorflow
(without output)
single GPU: 24s

## CPU C++ 
OMP implementation
partially optimised: 1 threads, 233.100s
partially optimised: 2 threads, 158.569s
partially optimised: 4 threads, 123.158s
partially optimised: 8 threads, 93.161s
partially optimised: 16 threads, 69.543s
partially optimised: 32 threads, 66.219s 

MPI implementation 

## Numpy 

range of different cores

## References
