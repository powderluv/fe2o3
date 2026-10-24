//! Full shared vecadd qualification data and independent dense F32 oracle.
//! This leaf imports no typed kernel or runtime and supplies no execution authority.

pub(crate) struct Case {
    pub(crate) name: String,
    pub(crate) c_len: usize,
    pub(crate) a: Vec<f32>,
    pub(crate) b: Vec<f32>,
}
pub(crate) fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    for length in [
        0_usize, 1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025,
    ] {
        all.push(Case {
            name: format!("nonquarter-{length}"),
            c_len: length,
            a: (0..length)
                .map(|i| ((i % 29) as i32 - 14) as f32 / 7.0)
                .collect(),
            b: (0..length)
                .map(|i| ((i % 31) as i32 - 15) as f32 / 11.0)
                .collect(),
        });
    }
    all.push(Case {
        name: "floating-specials-and-rounding".into(),
        c_len: 13,
        a: vec![
            0.0,
            -0.0,
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            1.0,
            f32::MAX,
            -f32::MAX,
            16_777_216.0,
            -16_777_216.0,
        ],
        b: vec![
            0.0,
            -0.0,
            0.0,
            -0.0,
            2.0,
            -2.0,
            f32::NEG_INFINITY,
            1.0,
            f32::NAN,
            f32::MAX,
            -f32::MAX,
            1.0,
            -1.0,
        ],
    });
    all.push(Case {
        name: "distinct-sufficient-input-extents".into(),
        c_len: 2,
        a: vec![1.0 / 7.0, -3.0 / 11.0, 99.0],
        b: vec![2.0 / 13.0, 1.0 / 17.0, 100.0, 200.0],
    });
    all
}

pub(crate) fn oracle(case: &Case) -> Result<Vec<f32>, std::io::Error> {
    if case.a.len() < case.c_len || case.b.len() < case.c_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "vecadd output extent exceeds an input",
        ));
    }
    Ok(case
        .a
        .iter()
        .zip(&case.b)
        .take(case.c_len)
        .map(|(a, b)| a + b)
        .collect())
}
