CXX = dpcpp
CXXFLAGS = -Wall -std=c++20
SYCLFLAGS = -fsycl
IFLAGS = -I ./include
LFLAGS = -L ./test/target/release -lblake3_test -lpthread

all: test_blake3

test/target/release/libblake3_test.a: test/src/lib.rs
	cd test; cargo build --release --lib -q

test/a.out: test/src/main.cpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $^ $(IFLAGS) $(LFLAGS) -o $@

test_blake3: test/target/release/libblake3_test.a test/a.out
	./test/a.out

clean:
	cd test; cargo clean
	find . -name 'a.out' -o -name '*.o' | xargs rm -f

bench/a.out: bench/main.cpp include/bench_blake3.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(IFLAGS) $< -o $@

benchmark: bench/a.out
	./bench/a.out
