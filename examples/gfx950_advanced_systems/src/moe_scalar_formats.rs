//! Scalar OCP E2M1/E4M3 decoding expressions for the real stored lowp bytes.
//! E4M3 follows the existing D decoder; this grants no source or proof authority.
#![allow(unused_macros)]
macro_rules! moe_decode_fp4 {
    ($value:expr) => {{
        let bits = $value;
        let exponent = (bits >> 1_u8) & 3;
        let mantissa = (bits & 1) as f32;
        let magnitude = if exponent == 0 { mantissa * 0.5 }
            else if exponent == 1 { 1.0 + mantissa * 0.5 }
            else if exponent == 2 { 2.0 + mantissa }
            else { 4.0 + 2.0 * mantissa };
        if bits & 8 == 0 { magnitude } else { -magnitude }
    }};
}
macro_rules! moe_decode_fp8 {
    ($value:expr) => {{
        let bits = $value;
        let exponent = (bits >> 3_u8) & 15;
        let mantissa = bits & 7;
        let magnitude = if exponent == 15 && mantissa == 7 {
            let nan_source = (mantissa ^ 7_u8) as f32;
            nan_source / nan_source
        } else if exponent == 0 { mantissa as f32 / 512.0 }
        else {
            let exponent0 = (exponent & 1) as f32;
            let exponent1 = ((exponent >> 1_u8) & 1) as f32;
            let exponent2 = ((exponent >> 2_u8) & 1) as f32;
            let exponent3 = ((exponent >> 3_u8) & 1) as f32;
            let scale = (1.0 + exponent0) * (1.0 + 3.0 * exponent1)
                * (1.0 + 15.0 * exponent2) * (1.0 + 255.0 * exponent3) / 128.0;
            (1.0 + mantissa as f32 / 8.0) * scale
        };
        if bits & 128 == 0 { magnitude } else { -magnitude }
    }};
}
