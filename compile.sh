#! /bin/bash
g++ -O3 src/lbm.cpp -I ./libs/eigen -o lbm -fopenmp
# ./a.out
# gnuplot plotting/plot.gnu 

# compiling in macos (clang compiler).
g++ -O3 src/lbm.cpp -I ./libs/eigen -o lbm -fopenmp
