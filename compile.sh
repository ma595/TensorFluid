#! /bin/bash
g++-13 -O3 src/lbm.cpp -I ./libs/eigen -o lbm -fopenmp
# ./a.out
# gnuplot plotting/plot.gnu 

# compiling in macos (clang compiler).
g++-13 -O3 src/lbm.cpp -I ./libs/eigen -o lbm -fopenmp
