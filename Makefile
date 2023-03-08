
CPP    := g++
CFLAGS := -O2 -Wall -std=c++20 --pedantic -fPIC
MPRSUP ?= #-DMPREAL_SUPPORT
LIBMPR ?= #-lmpfr

pls.o: pls.cpp pls.h
	$(CPP) $(CFLAGS) -c -I. $(MPRSUP) $< -o $@ $(LIBMPR)

pls: pls_main.cpp pls.o
	$(CPP) $(CFLAGS) $< -o $@

test: test.cpp 
	g++ $(CFLAGS) $(INCLUDE) test.cpp utility.cpp -o test 

clean:
	rm pls.o pls