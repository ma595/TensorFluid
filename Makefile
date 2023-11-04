CPP = g++-13
CPPFLAGS = -I/usr/local/opt/libomp/include -fopenmp -I/Users/matt/libs/ -O3
LDFLAGS = -L/usr/local/opt/libomp/lib

lbm: src/lbm.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)
