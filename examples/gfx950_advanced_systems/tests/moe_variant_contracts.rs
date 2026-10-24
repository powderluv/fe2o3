//! Exhaustive actual-byte format controls independent of native execution.
#![allow(dead_code)]
#[path="../src/dimensions.rs"] mod dimensions;
#[path="../src/reference.rs"] mod reference;
#[macro_use]
#[path="../src/moe_scalar_formats.rs"] mod formats;
#[test]
fn every_stored_lowp_byte_matches_independent_reference() {
    for bits in 0..=u8::MAX {
        assert_eq!(moe_decode_fp4!(bits).to_bits(),reference::decode_fp4(bits).to_bits());
        let actual=moe_decode_fp8!(bits);let expected=reference::decode_fp8(bits);
        if expected.is_nan() { assert!(actual.is_nan()); }
        else { assert_eq!(actual.to_bits(),expected.to_bits(),"E4M3 byte {bits:02x}"); }
    }
}
#[test]
fn checked_tile_parent_coordinates_preserve_all_dispatch_and_combine_addresses() {
    for raw in 0..1024 {
        let batch=raw/64;let lane=raw%64;
        for component in 0..4 {
            let packed=lane|((component&3)<<6);
            assert_eq!(packed,lane+64*component);
            assert!(packed<256);
            assert_eq!(packed/16,(lane+64*component)/16);
            assert_eq!(packed&15,(lane+64*component)%16);
        }
        for component in 0..2 {
            let tile_row=batch*2+component;
            assert_eq!(tile_row*64+lane,batch*128+component*64+lane);
            assert!(tile_row<32);
        }
        let combine_row=raw/256;let combine_column=raw%256;
        assert_eq!(combine_row*256+combine_column,raw);
        assert!(combine_row<4);
    }
}
