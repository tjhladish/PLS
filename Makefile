
.FORCE:

# use `make install TESTING="--prefix _install"` to dry run installation
TESTING ?=

install: build-static build-shared
	cmake --install build-shared $(TESTING)
	cmake --install build-static $(TESTING)

build-shared: .FORCE
	mkdir -p $@ && cd $@ && cmake .. -DBUILD_SHARED_LIBS=YES -DCMAKE_BUILD_TYPE=Release && cmake --build .

build-static: .FORCE
	mkdir -p $@ && cd $@ && cmake .. -DBUILD_SHARED_LIBS=NO -DCMAKE_BUILD_TYPE=Release && cmake --build .

clean: .FORCE
	git clean -ifdx