CXX = dpcpp
CXXFLAGS = -Wall -std=c++20
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
OPTFLAGS = -O3
IFLAGS = -I ./include
LFLAGS = -L ./test/target/release -lblake3_test -lpthread
BLAKE3_SIMD_LANES_FLAGS = -DBLAKE3_SIMD_LANES=$(shell echo $(or $(BLAKE3_SIMD_LANES),4)) # allowed values from set {2, 4, 8, 16}; default set to 4

all: test_blake3

test/target/release/libblake3_test.a: test/src/lib.rs
	cd test; cargo build --release --lib -q

test/a.out: test/src/main.cpp include/blake3.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) $< $(IFLAGS) $(LFLAGS) -o $@

test_blake3: test/target/release/libblake3_test.a test/a.out
	./test/a.out

clean:
	cd test; cargo clean
	find . -name 'a.out' -o -name '*.o' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla

bench/a.out: bench/main.cpp include/bench_blake3.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) $< -o $@

benchmark: bench/a.out
	./bench/a.out

aot_cpu:
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=avx512" bench/main.cpp -o bench/a.out; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=avx2" bench/main.cpp -o bench/a.out; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=avx" bench/main.cpp -o bench/a.out; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) -fsycl-targets=spir64_x86_64 -Xs "-march=sse4.2" bench/main.cpp -o bench/a.out; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi
	./bench/a.out

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) -fsycl-targets=spir64_gen -Xs "-device 0x4905" bench/main.cpp -o bench/a.out
	./bench/a.out

cuda:
	clang++ $(CXXFLAGS) $(SYCLFLAGS) $(OPTFLAGS) $(SYCLCUDAFLAGS) $(IFLAGS) $(BLAKE3_SIMD_LANES_FLAGS) bench/main.cpp -o bench/a.out
	./bench/a.out
