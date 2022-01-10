FROM intel/oneapi-basekit
# update system and install git ( to be used for cloning `blake3` repo )
RUN apt-get update; apt-get install -y git
# installing rustup; (nightly) toolchain will be later
RUN curl https://sh.rustup.rs -sSf | bash -s -- --default-toolchain none -y
# making rust toolchain available from anywhere inside container
ENV PATH="/root/.cargo/bin:${PATH}"
# required for running test cases; see https://github.com/itzmeanjan/blake3/blob/6516f00/test/rust-toolchain
RUN rustup toolchain install nightly-2021-12-04
WORKDIR /home
# clone target repo's `master` branch inside container and
# prepare a bash script which will be used for running test cases
#
# when this image is run all possible combinations are tested i.e. 2, 4, 8, 16
# chunks are compressed together and results of both SYCL blake3 implementations
# are asserted against original Rust implementation of Blake3
RUN git clone -b master https://github.com/itzmeanjan/blake3.git; echo "cd blake3; \
    BLAKE3_SIMD_LANES=2 make; make clean; \
    BLAKE3_SIMD_LANES=4 make; make clean; \
    BLAKE3_SIMD_LANES=8 make; make clean; \
    BLAKE3_SIMD_LANES=16 make; make clean" > run_test
CMD ["bash", "run_test"]
