CFLAGS=--ansi --pedantic -std=c++0x -O2
#CFLAGS=--ansi --pedantic -std=c++0x -g 
INCLUDE= -I.
all: pls

pls: pls.cpp 
	g++ $(CFLAGS) $(INCLUDE) pls.cpp utility.cpp -o pls

pls_mpreal: pls.cpp 
	g++ $(CFLAGS) $(INCLUDE) -DMPREAL_SUPPORT pls.cpp utility.cpp -o pls_mpreal -lmpfr 

test: test.cpp 
	g++ $(CFLAGS) $(INCLUDE) test.cpp utility.cpp -o test 

clean:
	rm pls pls_mpreal test
