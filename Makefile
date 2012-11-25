CFLAGS=--ansi --pedantic -std=c++0x -O2
#CFLAGS=--ansi --pedantic -std=c++0x -g 
#INCLUDE= -I/lib/eigen/unsupported/test/mpreal/ -I/home/tjhladish/work/lib/eigen/ 
INCLUDE= -I/home/tjhladish/work/lib/eigen/ 
all: pls

pls: pls.cpp 
	g++ $(CFLAGS) $(INCLUDE) pls.cpp Utility.cpp -o pls

pls_mpreal: pls.cpp 
	g++ $(CFLAGS) $(INCLUDE) pls.cpp Utility.cpp -o pls_mpreal -lmpfr 

test: test.cpp 
	g++ $(CFLAGS) $(INCLUDE) test.cpp Utility.cpp -o test 

clean:
	rm pls pls_mpreal test
