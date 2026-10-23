//! Mathematical source controls; these do not qualify device execution.
#![allow(dead_code)]
#[path = "../src/dimensions.rs"]
mod dimensions;
#[path = "../src/reference.rs"]
mod reference;

#[test]
fn bounded_route_table_matches_independent_ocp_decoder_for_every_byte() {
    for bits in 0..=u8::MAX {
        let shift = ((bits & 7) as u32) * 4;
        assert!(shift <= 28);
        let packed = if shift < 32 { 0xc864_3210_u32 >> shift } else { 0 };
        let magnitude = (packed & 15) as f32 * 0.5;
        let value = (1.0 - 2.0 * ((bits >> 3) & 1) as f32) * magnitude;
        assert_eq!(value.to_bits(), reference::decode_fp4(bits).to_bits());
    }
}

#[test]
fn bounded_packed_records_match_independent_byte_extraction() {
    for packed in [0_u64, u64::MAX, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210,
                   0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555] {
        let bytes = packed.to_le_bytes();
        for record in 0..dimensions::TOKENS * dimensions::TOP_K {
            let shift = 2 * record;
            assert!(shift <= 62);
            let expected = (bytes[record / 4] >> (2 * (record % 4))) & 3;
            let selected = if shift < 64 { ((packed >> shift) & 3) as u32 } else { 0 };
            assert_eq!(selected as u8, expected);
            assert_eq!(packed >> shift, packed.wrapping_shr(shift as u32));
        }
    }
}
