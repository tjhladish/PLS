
CPP    := g++
CFLAGS := -O2 -Wall -std=c++20 --pedantic -fPIC
MPRSUP ?= #-DMPREAL_SUPPORT
LIBMPR ?= #-lmpfr

ARCHIVE ?= $(AR) -rv

pls.o: pls.cpp pls.h
	$(CPP) $(CFLAGS) -c -I. -Ieigen $(MPRSUP) $< -o $@ $(LIBMPR)

libpls.a: pls.o
	$(ARCHIVE) $@ $^

pls: pls_main.cpp libpls.a
	$(CPP) $(CFLAGS) -I. -Ieigen -o $@ $< -L. -lpls $(LIBMPR)

clean:
	rm pls.o pls