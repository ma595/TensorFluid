CPP = /usr/local/opt/llvm/bin/clang++
CPPFLAGS = -I/usr/local/opt/llvm/include -fopenmp -I/Users/matt/libs/ -O3
LDFLAGS = -L/usr/local/opt/llvm/lib

lbm: src/lbm.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)
