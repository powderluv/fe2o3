use std::convert::Infallible;
use std::ffi::OsString;

use super::super::*;

const ONE: &str = r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"}]}"#;
const TWO: &str =
    r#"{"bindings":[{"reference":"r","kernel":"a"},{"kernel":"b","reference":"s"}],"version":1}"#;

fn compare(text: &str) {
    let full = ReferenceEnrollmentRequestV1::decode(text, |_| Ok::<_, Infallible>(()))
        .map(|request| request.bindings().len());
    let projected =
        ReferenceEnrollmentRequestV1::project_binding_count(text, |_| Ok::<_, Infallible>(()));
    assert_eq!(projected, full, "{text:?}");
}

fn env(entries: &[(&str, &str)]) -> CompileEnvironmentV2 {
    CompileEnvironmentV2::from_child_environment(
        entries
            .iter()
            .map(|&(k, v)| (OsString::from(k), OsString::from(v))),
    )
    .unwrap()
}

#[test]
fn projection_checks_closed_schema_and_semantic_precedence() {
    for text in [
        ONE,
        TWO,
        "",
        "null",
        "[]",
        "{}",
        r#"{"version":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[["a","r"]]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r","extra":0}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","kernel":"b","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r","refer\u0065nce":"s"}]}"#,
        r#"{"version":1,"version":1,"bindings":[]}"#,
        r#"{"version":1,"vers\u0069on":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[],"bindings":[]}"#,
        r#"{"version":1,"bindings":null}"#,
        r#"{"version":1,"bindings":[null]}"#,
        r#"{"version":2,"bindings":[]} false"#,
        r#"{"version":2,"bindings":[{"kernel":"","reference":"r"}],"extra":0}"#,
        r#"{"version":2,"bindings":[{"kernel":"","reference":"r"},["a","r"]]}"#,
        r#"{"version":1,"bindings":[{"kernel":"","reference":"r"},{"kernel":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"b","reference":"r"},{"kernel":"a","reference":""}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"},{"kernel":"\u0061","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"b","reference":"r"},{"kernel":"a","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"},]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r",}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"}],}"#,
    ] {
        compare(text);
    }
    for version in [
        "-0", "-1", "1.0", "1e0", "65536", "1e400", "null", "true", "[]", "{}", "0", "65535", "01",
        "1 0", "+1",
    ] {
        compare(&format!(r#"{{"version":{version},"bindings":[]}}"#));
    }
    for tail in ["", " ", "\t\r\n", "\u{a0}", "{}", "null", ",", "\0"] {
        compare(&format!("{ONE}{tail}"));
    }
}

#[test]
fn projection_compares_decoded_unicode_not_json_spelling() {
    for kernel in [
        r"a\u0062",
        r#"a\"b"#,
        r"a\\b",
        r"a\/b",
        r"\ud83d\ude80",
        r"\udbff\udfff",
        r"\u0000",
        r"\u0085",
        r"\ud800",
        r"\udc00",
        r"\ud800\u0000",
        r"\ud800 \udc00",
        r"\u12g4",
        r"\x41",
        r"\u",
        r"\b",
        r"\f",
        r"\n",
        r"\r",
        r"\t",
    ] {
        compare(&format!(
            r#"{{"vers\u0069on":1,"bind\u0069ngs":[{{"kern\u0065l":"{kernel}","refer\u0065nce":"r"}}]}}"#
        ));
    }
    for scalar in [
        0, 1, 8, 31, 32, 34, 47, 92, 127, 128, 159, 160, 0x7ff, 0x800, 0xd7ff, 0xe000, 0xffff,
        0x10000, 0x1f680, 0x10ffff,
    ] {
        let ch = char::from_u32(scalar).unwrap();
        let escaped = if scalar <= 0xffff {
            format!(r"\u{scalar:04x}")
        } else {
            let n = scalar - 0x10000;
            format!(
                r"\u{:04x}\u{:04x}",
                0xd800 + (n >> 10),
                0xdc00 + (n & 0x3ff)
            )
        };
        let raw = serde_json::to_string(&ch.to_string()).unwrap();
        for pair in [
            format!(r#"{raw},"{escaped}""#),
            format!(r#""{escaped}",{raw}"#),
        ] {
            let (a, b) = pair.split_once(',').unwrap();
            compare(&format!(
                r#"{{"version":1,"bindings":[{{"kernel":{a},"reference":"r"}},{{"kernel":{b},"reference":"r"}}]}}"#
            ));
        }
        compare(&format!(
            r#"{{"version":1,"bindings":[{{"kernel":"{ch}","reference":"r"}}]}}"#
        ));
    }
    // UTF-8 and scalar lexicographic ordering agree across encoding boundaries.
    compare(
        r#"{"version":1,"bindings":[{"kernel":"\u007f","reference":"r"},{"kernel":"\u0080","reference":"r"}]}"#,
    );
    compare(
        r#"{"version":1,"bindings":[{"kernel":"\ue000","reference":"r"},{"kernel":"\ud800\udc00","reference":"r"}]}"#,
    );
}

#[test]
fn projection_checks_byte_selector_and_roster_bounds() {
    for len in [0, 1, 1023, 1024, 1025] {
        for selector in [
            "x".repeat(len),
            "\u{e9}".repeat(len / 2),
            r"\u0061".repeat(len),
        ] {
            let text = format!(
                r#"{{"version":1,"bindings":[{{"kernel":"{selector}","reference":"r"}}]}}"#
            );
            compare(&text);
        }
    }
    for n in [0, 1, 2, 64, 128, 256, 257] {
        let bindings: Vec<_> = (0..n)
            .map(|i| serde_json::json!({"kernel":format!("k{i:04}"),"reference":"r"}))
            .collect();
        compare(&serde_json::json!({"version":1,"bindings":bindings}).to_string());
    }
    for len in [4095, 4096, 4097] {
        compare(&format!("{ONE}{}", " ".repeat(len - ONE.len())));
    }
    compare(&format!(
        r#"{{"version":1,"bindings":[],"{}":0}}"#,
        "x".repeat(2048)
    ));
    compare(&format!(
        r#"{{"version":1,"bindings":[{{"kernel":{}0{},"reference":"r"}}]}}"#,
        "[".repeat(256),
        "]".repeat(256)
    ));
}

#[test]
fn projection_matches_decoder_for_all_single_ascii_edits() {
    for seed in [ONE, TWO] {
        for i in 0..seed.len() {
            compare(&format!("{}{}", &seed[..i], &seed[i + 1..]));
            compare(&seed[..i]);
            for byte in 0..=127 {
                let ch = char::from(byte);
                compare(&format!("{}{ch}{}", &seed[..i], &seed[i + 1..]));
                compare(&format!("{}{ch}{}", &seed[..i], &seed[i..]));
            }
        }
    }
}

#[test]
fn projection_preserves_charges_absence_and_captured_environment() {
    let environments = [
        env(&[]),
        env(&[("AAA", "not-json"), ("ZZZ", ONE)]),
        env(&[
            ("AAA", "ignored"),
            (REFERENCE_ENROLLMENT_ENV_V1, TWO),
            ("ZZZ", "not-json"),
        ]),
        env(&[(REFERENCE_ENROLLMENT_ENV_V1, "not-json")]),
    ];
    for environment in environments {
        let (mut full_charges, mut projection_charges) = (Vec::new(), Vec::new());
        let full = ReferenceEnrollmentRequestV1::from_environment(&environment, |n| {
            full_charges.push(n);
            Ok::<_, Infallible>(())
        })
        .map(|v| v.map(|r| r.bindings().len()));
        let projection = ReferenceEnrollmentRequestV1::project_binding_count_from_environment(
            &environment,
            |n| {
                projection_charges.push(n);
                Ok::<_, Infallible>(())
            },
        );
        assert_eq!(projection, full);
        assert_eq!(projection_charges, full_charges);
    }
    for stop in 1..=3 {
        let mut calls = 0;
        let result = ReferenceEnrollmentRequestV1::project_binding_count_from_environment(
            &env(&[(REFERENCE_ENROLLMENT_ENV_V1, "not-json")]),
            |_| {
                calls += 1;
                if calls == stop { Err(stop) } else { Ok(()) }
            },
        );
        assert_eq!(calls, stop);
        assert_eq!(result, Err(ReferenceEnrollmentDecodeErrorV1::Work(stop)));
    }
    let mut charges = Vec::new();
    let result = ReferenceEnrollmentRequestV1::project_binding_count(&" ".repeat(4097), |n| {
        charges.push(n);
        Ok::<_, Infallible>(())
    });
    assert!(result.is_err());
    assert_eq!(charges, [1]);
}

#[test]
fn projected_count_changes_with_original_request_not_a_header() {
    assert_eq!(
        ReferenceEnrollmentRequestV1::project_binding_count(ONE, |_| Ok::<_, Infallible>(())),
        Ok(1)
    );
    assert_eq!(
        ReferenceEnrollmentRequestV1::project_binding_count(TWO, |_| Ok::<_, Infallible>(())),
        Ok(2)
    );
    compare(&ONE.replace("\"reference\":\"r\"", "\"count\":1"));
    compare(&ONE.replace("\"version\":1", "\"version\":1,\"count\":1"));
}
