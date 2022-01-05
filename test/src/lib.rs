#![feature(vec_into_raw_parts)]

extern crate blake3;

#[no_mangle]
pub unsafe extern "C" fn rust_blake3(input: *mut u8, i_size: u64, digest: *mut u8) -> *mut u8 {
    let input_ = {
        assert!(!input.is_null());
        Vec::from_raw_parts(input, i_size as usize, i_size as usize)
    };

    let hash = blake3::hash(&input_);

    let mut digest_ = {
        assert!(!digest.is_null());
        Vec::from_raw_parts(digest, 32, 32)
    };

    digest_[..32].copy_from_slice(&hash.as_bytes()[..]);
    let (ptr, _, _) = digest_.into_raw_parts();
    ptr
}
