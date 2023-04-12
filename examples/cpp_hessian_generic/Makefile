EXAMPLE = msa 

all: test

CXX 		:= /u/drehwald/prog/llvm-project/build/bin/clang++
CC 		:= /u/drehwald/prog/llvm-project/build/bin/clang
LLDEnzyme 	:= /u/drehwald/prog/Enzyme/enzyme/build/Enzyme/LLDEnzyme-14.so


CXXFLAGS 	+= -O3 -fuse-ld=lld -flto
LDFLAGS 	+= -fuse-ld=lld -flto
EnzymeNoOpt 	:= -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

# Eigen is included by default


test: main.cpp 
	$(CXX) $(CXXFLAGS) -c $(EnzymeNoOpt) main.cpp
	echo "Compiling done, now linking\n\n"
	$(CXX) $(CXXFLAGS) main.o $(LDFLAGS) -o tFS_t -Wl,--lto-legacy-pass-manager -Wl,-mllvm=-load=$(LLDEnzyme) -Wl,-mllvm=-enzyme-strict-aliasing=0 -Wl,-mllvm=-enzyme-loose-types
	echo "done"
