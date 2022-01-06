#![feature(vec_into_raw_parts)]

extern crate blake3;

#[no_mangle]
pub unsafe extern "C" fn rust_blake3(input: *mut u8, i_size: u64) -> *mut u8 {
    let input_ = {
        assert!(!input.is_null());
        Vec::from_raw_parts(input, i_size as usize, i_size as usize)
    };

    // compute blake3 hash on provided input
    let hash = blake3::hash(&input_);

    // heap allocate memory for storing blake3 digest
    let mut digest = vec![0; 32];
    digest[..32].copy_from_slice(&hash.as_bytes()[..]);

    // return address to allocated memory for consumption on caller side
    let (ptr, _, _) = digest.into_raw_parts();
    ptr
}
