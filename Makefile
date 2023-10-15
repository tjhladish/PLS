
.FORCE:

# use `make install TESTING="--prefix _install"` to dry run installation
TESTING ?=

install: build-static build-shared
	cmake --install build-shared $(TESTING)
	cmake --install build-static $(TESTING)

build: .FORCE
	mkdir -p $@ && cd $@ && cmake ..

build-shared: .FORCE
	mkdir -p $@ && cd $@ && cmake .. -DBUILD_SHARED_LIBS=YES -DCMAKE_BUILD_TYPE=Release && cmake --build .

build-static: .FORCE
	mkdir -p $@ && cd $@ && cmake .. -DBUILD_SHARED_LIBS=NO -DCMAKE_BUILD_TYPE=Release && cmake --build .

build-debug: .FORCE
	mkdir -p $@ && cd $@ && cmake .. -DBUILD_SHARED_LIBS=NO -DCMAKE_BUILD_TYPE=Debug && cmake --build .

valgrind: build-debug
	cd $^ && valgrind --leak-check=yes ./PLS ../toyX.csv ../toyY.csv 2

clean: .FORCE
	git clean -ifdx