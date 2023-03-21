
.FORCE:

build: .FORCE
	cmake -S . -B $@

build-shared: .FORCE
	cmake -S . -B $@ -DBUILD_SHARED_LIBS=YES -DCMAKE_BUILD_TYPE=Release
	cmake --build $@

build-static: .FORCE
	cmake -S . -B $@ -DBUILD_SHARED_LIBS=NO -DCMAKE_BUILD_TYPE=Release
	cmake --build $@

install: build-static build-shared
	cmake --install build-shared
	cmake --install build-static