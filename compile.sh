#! /bin/bash
g++ -O3 lbm.cpp -I ./libs/eigen -o lbm -fopenmp
# ./a.out
# gnuplot plotting/plot.gnu 
