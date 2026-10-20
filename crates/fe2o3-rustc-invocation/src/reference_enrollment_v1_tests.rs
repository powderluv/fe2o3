use std::convert::Infallible;
use std::ffi::OsString;
use std::mem::size_of;

use super::*;

const REQUEST: &str =
    r#"{"version":1,"bindings":[{"kernel":"demo::write","reference":"demo::spec"}]}"#;

fn decode(
    text: &str,
) -> Result<ReferenceEnrollmentRequestV1, ReferenceEnrollmentDecodeErrorV1<Infallible>> {
    ReferenceEnrollmentRequestV1::decode(text, |_| Ok(()))
}

fn environment(entries: &[(&str, &str)]) -> CompileEnvironmentV2 {
    CompileEnvironmentV2::from_child_environment(
        entries
            .iter()
            .map(|&(key, value)| (OsString::from(key), OsString::from(value))),
    )
    .unwrap()
}

#[test]
fn selectors_are_read_only_descriptions_with_no_name_matching_requirement() {
    let request = decode(REQUEST).unwrap();
    assert_eq!(request.version(), 1);
    assert_eq!(request.bindings()[0].kernel(), "demo::write");
    assert_eq!(request.bindings()[0].reference(), "demo::spec");
    let request = decode(
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"spec"},{"kernel":"b","reference":"spec"}]}"#,
    ).unwrap();
    assert_eq!(request.bindings().len(), 2);
    assert_eq!(request.bindings()[1].reference(), "spec");
}

#[test]
fn rejects_unknown_duplicate_missing_fields_and_non_object_shapes() {
    for text in [
        r#"[1,[{"kernel":"a","reference":"b"}]]"#,
        r#"{"version":1,"bindings":[["a","b"]]}"#,
        r#"{"version":1,"version":1,"bindings":[]}"#,
        r#"{"version":1,"vers\u0069on":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[],"bindings":[]}"#,
        r#"{"version":1,"bindings":[],"authority":true}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","kernel":"a","reference":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"b","reference":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"b","proof":true}]}"#,
        r#"{"bindings":[{"kernel":"a","reference":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a"}]}"#,
        r#"{"version":1,"bindings":null}"#,
        r#"{"version":1,"bindings":[]} null"#,
        r#"{"version":"1","bindings":[]}"#,
        r#"{"version":1,"bindings":[null]}"#,
    ] {
        assert_eq!(
            decode(text).unwrap_err().to_string(),
            "invalid reference enrollment request schema",
            "{text}"
        );
    }
}

#[test]
fn version_empty_roster_and_unsorted_or_duplicate_roots_keep_distinct_errors() {
    assert_eq!(
        decode(&REQUEST.replace("\"version\":1", "\"version\":2"))
            .unwrap_err()
            .to_string(),
        "unsupported reference enrollment request version"
    );
    assert_eq!(
        decode(r#"{"version":1,"bindings":[]}"#)
            .unwrap_err()
            .to_string(),
        "reference enrollment binding count outside limits"
    );
    for keys in [["a", "a"], ["b", "a"]] {
        let text = serde_json::json!({"version":1,"bindings":[
            {"kernel":keys[0],"reference":"r"}, {"kernel":keys[1],"reference":"r"}
        ]})
        .to_string();
        assert_eq!(
            decode(&text).unwrap_err().to_string(),
            "reference enrollment roots must be unique and sorted"
        );
    }
    assert_eq!(
        decode(r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"},{"kernel":"\u0061","reference":"r"}]}"#)
            .unwrap_err().to_string(),
        "reference enrollment roots must be unique and sorted"
    );
}

#[test]
fn selector_limits_count_decoded_utf8_bytes_and_reject_controls() {
    for value in [
        String::new(),
        "x".repeat(MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1 + 1),
        "\u{e9}".repeat(MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1 / 2 + 1),
        "a\nb".to_owned(),
        "\u{85}".to_owned(),
    ] {
        for field in ["kernel", "reference"] {
            let mut json: serde_json::Value = serde_json::from_str(REQUEST).unwrap();
            json["bindings"][0][field] = value.clone().into();
            assert_eq!(
                decode(&json.to_string()).unwrap_err().to_string(),
                "invalid reference enrollment selector"
            );
        }
    }
    let mut json: serde_json::Value = serde_json::from_str(REQUEST).unwrap();
    for field in ["kernel", "reference"] {
        json["bindings"][0][field] = "\u{e9}"
            .repeat(MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1 / 2)
            .into();
    }
    assert!(decode(&json.to_string()).is_ok());
    assert_eq!(
        decode(r#"{"version":1,"bindings":[{"kernel":"a\u0062","reference":"r"}]}"#)
            .unwrap()
            .bindings()[0]
            .kernel(),
        "ab"
    );
}

#[test]
fn encoded_byte_bound_precedes_roster_and_schema_validation() {
    assert_eq!(MAX_REFERENCE_ENROLLMENT_BYTES_V1, 4096);
    assert_eq!(MAX_REFERENCE_ENROLLMENT_BINDINGS_V1, 256);
    assert_eq!(MAX_REFERENCE_ENROLLMENT_SELECTOR_BYTES_V1, 1024);
    let padded = format!(
        "{REQUEST}{}",
        " ".repeat(MAX_REFERENCE_ENROLLMENT_BYTES_V1 - REQUEST.len())
    );
    assert!(decode(&padded).is_ok());
    assert_eq!(
        decode(&(padded + " ")).unwrap_err().to_string(),
        "reference enrollment request exceeds byte limit"
    );
    let rows = |count| {
        serde_json::json!({"version":1,"bindings":
            (0..count).map(|i| serde_json::json!({"kernel":format!("k{i:04}"),"reference":"r"}))
                .collect::<Vec<_>>()
        })
        .to_string()
    };
    assert!(decode(&rows(64)).is_ok());
    assert_eq!(
        decode(&rows(MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 + 1))
            .unwrap_err()
            .to_string(),
        "reference enrollment request exceeds byte limit"
    );
    let mut charges = Vec::new();
    let error = ReferenceEnrollmentRequestV1::decode(
        &" ".repeat(MAX_REFERENCE_ENROLLMENT_BYTES_V1 + 1),
        |amount| {
            charges.push(amount);
            Ok::<_, Infallible>(())
        },
    )
    .unwrap_err();
    assert_eq!(charges, [1]);
    assert_eq!(
        error.to_string(),
        "reference enrollment request exceeds byte limit"
    );
}

#[test]
fn decoder_prepays_the_exact_quote_before_schema_validation() {
    for text in [REQUEST, "not-json"] {
        let mut charges = Vec::new();
        let result = ReferenceEnrollmentRequestV1::decode(text, |amount| {
            charges.push(amount);
            Ok::<_, Infallible>(())
        });
        assert_eq!(
            charges,
            [
                1,
                text.len() * 64 + std::mem::size_of::<ReferenceEnrollmentRequestV1>()
            ]
        );
        assert_eq!(result.is_ok(), text == REQUEST);
    }
}

#[test]
fn environment_scan_stops_at_the_request_and_keeps_charges_cumulative() {
    let captured = environment(&[
        ("AAA", "ignored"),
        (REFERENCE_ENROLLMENT_ENV_V1, REQUEST),
        ("ZZZ", "not-json"),
    ]);
    let mut total = 17;
    let mut charges = Vec::new();
    for _ in 0..2 {
        let request = ReferenceEnrollmentRequestV1::from_environment(&captured, |amount| {
            total += amount;
            charges.push(amount);
            Ok::<_, Infallible>(())
        })
        .unwrap()
        .unwrap();
        assert_eq!(request, decode(REQUEST).unwrap());
    }
    let once = [
        4,
        REFERENCE_ENROLLMENT_ENV_V1.len() + 1,
        1,
        REQUEST.len() * 64 + std::mem::size_of::<ReferenceEnrollmentRequestV1>(),
    ];
    assert_eq!(charges, once.repeat(2));
    assert_eq!(total, 17 + 2 * once.iter().sum::<usize>());
}

#[test]
fn absence_charges_only_captured_keys_without_decoding_other_values() {
    let captured = environment(&[("AAA", "not-json"), ("ZZZ", REQUEST)]);
    let mut charges = Vec::new();
    let result = ReferenceEnrollmentRequestV1::from_environment(&captured, |amount| {
        charges.push(amount);
        Ok::<_, Infallible>(())
    })
    .unwrap();
    assert!(result.is_none());
    assert_eq!(charges, [4, 4]);
    let mut called = false;
    assert!(
        ReferenceEnrollmentRequestV1::from_environment(&environment(&[]), |_| {
            called = true;
            Ok::<_, Infallible>(())
        })
        .unwrap()
        .is_none()
    );
    assert!(!called);
}

#[derive(Debug, Eq, PartialEq)]
struct Refused(usize);

#[test]
fn charge_refusals_are_returned_unchanged_before_decoding() {
    for stop in [1, 2] {
        let mut calls = 0;
        let error = ReferenceEnrollmentRequestV1::decode("not-json", |_| {
            calls += 1;
            if calls == stop {
                Err(Refused(stop))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(calls, stop);
        assert_eq!(error, ReferenceEnrollmentDecodeErrorV1::Work(Refused(stop)));
    }
    let captured = environment(&[(REFERENCE_ENROLLMENT_ENV_V1, "not-json")]);
    for stop in [1, 2, 3] {
        let mut calls = 0;
        let error = ReferenceEnrollmentRequestV1::from_environment(&captured, |_| {
            calls += 1;
            if calls == stop {
                Err(Refused(stop))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(calls, stop);
        assert_eq!(error, ReferenceEnrollmentDecodeErrorV1::Work(Refused(stop)));
    }
}

#[test]
fn closed_seed_keeps_schema_version_roster_selector_and_order_precedence() {
    let schema = "invalid reference enrollment request schema";
    for text in [
        r#"{"version":2,"bindings":[{"kernel":"","reference":"r"}],"extra":0}"#,
        r#"{"version":2,"bindings":[{"kernel":"","reference":"r"},["a","r"]]}"#,
        r#"{"version":2,"bindings":[]} false"#,
        r#"{"version":1,"bindings":[{"kernel":"","reference":"r"},{"kernel":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"","reference":"r"}"#,
    ] {
        assert_eq!(decode(text).unwrap_err().to_string(), schema, "{text}");
    }
    for text in [
        r#"{"version":2,"bindings":[]}"#,
        r#"{"bindings":[{"kernel":"","reference":"r"}],"version":2}"#,
    ] {
        assert_eq!(
            decode(text).unwrap_err().to_string(),
            "unsupported reference enrollment request version"
        );
    }
    assert_eq!(
        decode(r#"{"version":1,"bindings":[{"kernel":"b","reference":"r"},{"kernel":"a","reference":""}]}"#)
            .unwrap_err().to_string(),
        "invalid reference enrollment selector"
    );
}

#[test]
fn closed_seed_preserves_strict_u16_numeric_schema() {
    for version in [
        "-0", "-1", "1.0", "1e0", "65536", "1e400", "null", "true", "[]", "{}",
    ] {
        let text = format!(r#"{{"version":{version},"bindings":[]}}"#);
        assert_eq!(
            decode(&text).unwrap_err().to_string(),
            "invalid reference enrollment request schema",
            "{version}"
        );
    }
    for version in ["0", "65535"] {
        let text = format!(r#"{{"version":{version},"bindings":[]}}"#);
        assert_eq!(
            decode(&text).unwrap_err().to_string(),
            "unsupported reference enrollment request version"
        );
    }
}

#[test]
fn closed_seed_decodes_escaped_keys_and_surrogates_without_dynamic_key_owners() {
    let request = decode(
        r#"{"vers\u0069on":1,"bind\u0069ngs":[{"kern\u0065l":"a\ud83d\ude80","refer\u0065nce":"r\u00e9"}]}"#,
    ).unwrap();
    assert_eq!(request.bindings()[0].kernel(), "a\u{1f680}");
    assert_eq!(request.bindings()[0].reference(), "r\u{e9}");
    for text in [
        r#"{"version":1,"bindings":[{"kernel":"a","kern\u0065l":"b","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"\ud800","reference":"r"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"\udc00","reference":"r"}]}"#,
    ] {
        assert_eq!(
            decode(text).unwrap_err().to_string(),
            "invalid reference enrollment request schema"
        );
    }
}

#[test]
fn closed_seed_retains_one_bounded_vector_and_exact_selector_backing() {
    let request = decode(REQUEST).unwrap();
    assert_eq!(
        request.bindings.capacity(),
        MAX_REFERENCE_ENROLLMENT_BINDINGS_V1
    );
    let mut strings = 0;
    for row in &request.bindings {
        assert_eq!(row.kernel.capacity(), row.kernel.len());
        assert_eq!(row.reference.capacity(), row.reference.len());
        strings += row.kernel.capacity() + row.reference.capacity();
    }
    assert!(strings <= REQUEST.len());
    assert_eq!(
        request.retained_storage_bytes(),
        Some(
            size_of::<ReferenceEnrollmentRequestV1>()
                + MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 * size_of::<ReferenceEnrollmentBindingV1>()
                + strings
        )
    );
}

#[test]
fn closed_seed_defers_semantic_refusals_without_retaining_invalid_selectors() {
    for selector in ["x".repeat(1025), "\n".to_owned(), String::new()] {
        let text = serde_json::json!({
            "version": 1, "bindings": [{"kernel": selector, "reference": "r"}]
        })
        .to_string();
        let parsed = parse::request(&text).unwrap();
        assert!(parsed.invalid_selector);
        assert_eq!(parsed.binding_count, 1);
        assert_eq!(parsed.request.bindings[0].kernel.capacity(), 0);
        assert_eq!(parsed.request.bindings[0].reference, "r");
        assert_eq!(
            decode(&text).unwrap_err().to_string(),
            "invalid reference enrollment selector"
        );
    }
}

#[test]
fn closed_seed_never_grows_output_for_excess_rows_but_finishes_schema() {
    // Exercise the private schema component beyond the earlier public byte gate.
    let mut rows = (0..257)
        .map(|i| serde_json::json!({"kernel": format!("k{i:04}"), "reference": "r"}))
        .collect::<Vec<_>>();
    let text = serde_json::json!({"version": 1, "bindings": rows}).to_string();
    let parsed = parse::request(&text).unwrap();
    assert_eq!(parsed.binding_count, 257);
    assert_eq!(parsed.request.bindings.len(), 256);
    assert_eq!(parsed.request.bindings.capacity(), 256);
    rows[256] = serde_json::json!({"kernel": "z"});
    let invalid = serde_json::json!({"version": 1, "bindings": rows}).to_string();
    assert!(parse::request(&invalid).is_err());
    assert_eq!(
        decode(&text).unwrap_err().to_string(),
        "reference enrollment request exceeds byte limit"
    );
}

#[test]
fn closed_seed_keeps_static_public_schema_errors_for_long_and_nested_inputs() {
    for text in [
        format!(r#"{{"version":1,"bindings":[],"{}":0}}"#, "x".repeat(2048)),
        format!(
            r#"{{"version":1,"bindings":[{{"kernel":{}0{},"reference":"r"}}]}}"#,
            "[".repeat(256),
            "]".repeat(256)
        ),
        r#"{"version":1,"bindings":[{"kernel":false,"reference":"r"}]}"#.to_owned(),
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r","x":[}"#.to_owned(),
    ] {
        assert!(text.len() <= MAX_REFERENCE_ENROLLMENT_BYTES_V1);
        assert_eq!(
            decode(&text).unwrap_err().to_string(),
            "invalid reference enrollment request schema"
        );
    }
}

#[test]
fn retained_storage_counts_empty_and_spare_vector_capacity() {
    // Private synthetic owners exercise storage arithmetic, not schema admission.
    for bindings in [Vec::new(), Vec::with_capacity(13)] {
        let expected = size_of::<ReferenceEnrollmentRequestV1>()
            + bindings.capacity() * size_of::<ReferenceEnrollmentBindingV1>();
        let request = ReferenceEnrollmentRequestV1 {
            version: 1,
            bindings,
        };
        assert_eq!(request.retained_storage_bytes(), Some(expected));
        assert!(request.bindings().is_empty());
    }
}

#[test]
fn retained_storage_counts_capacity_of_empty_strings() {
    let mut bindings = Vec::with_capacity(7);
    bindings.push(ReferenceEnrollmentBindingV1 {
        kernel: String::with_capacity(17),
        reference: String::with_capacity(31),
    });
    let expected = size_of::<ReferenceEnrollmentRequestV1>()
        + bindings.capacity() * size_of::<ReferenceEnrollmentBindingV1>()
        + bindings[0].kernel.capacity()
        + bindings[0].reference.capacity();
    let request = ReferenceEnrollmentRequestV1 {
        version: 1,
        bindings,
    };
    assert_eq!(request.retained_storage_bytes(), Some(expected));
    assert!(request.bindings()[0].kernel().is_empty());
    assert!(request.bindings()[0].reference().is_empty());
}

#[test]
fn retained_storage_counts_every_decoded_binding_and_both_selectors() {
    let request = decode(
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"first"},{"kernel":"bb","reference":"second"}]}"#,
    ).unwrap();
    let expected = size_of::<ReferenceEnrollmentRequestV1>()
        + request.bindings.capacity() * size_of::<ReferenceEnrollmentBindingV1>()
        + request.bindings[0].kernel.capacity()
        + request.bindings[0].reference.capacity()
        + request.bindings[1].kernel.capacity()
        + request.bindings[1].reference.capacity();
    assert_eq!(request.retained_storage_bytes(), Some(expected));
    assert_eq!(request, decode(
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"first"},{"kernel":"bb","reference":"second"}]}"#,
    ).unwrap());
}

#[test]
fn retained_storage_tracks_each_spare_capacity_without_changing_descriptions() {
    let text = r#"{"version":1,"bindings":[{"kernel":"a","reference":"first"},{"kernel":"bb","reference":"second"}]}"#;
    for field in 0..5 {
        let mut request = decode(text).unwrap();
        let before = request.retained_storage_bytes().unwrap();
        let increase = if field == 0 {
            let old = request.bindings.capacity();
            request.bindings.reserve_exact(old + 17);
            (request.bindings.capacity() - old) * size_of::<ReferenceEnrollmentBindingV1>()
        } else {
            let binding = &mut request.bindings[(field - 1) / 2];
            let selector = if field % 2 == 1 {
                &mut binding.kernel
            } else {
                &mut binding.reference
            };
            let old = selector.capacity();
            selector.reserve_exact(old + 19);
            selector.capacity() - old
        };
        assert!(increase > 0);
        assert_eq!(request.retained_storage_bytes(), Some(before + increase));
        assert_eq!(request, decode(text).unwrap());
    }
}

#[test]
fn retained_storage_arithmetic_is_checked_without_large_allocations() {
    let header = size_of::<ReferenceEnrollmentRequestV1>();
    let binding_header = size_of::<ReferenceEnrollmentBindingV1>();
    assert_eq!(checked_retained_storage_bytes(0, []), Some(header));
    assert_eq!(
        checked_retained_storage_bytes(3, [(7, 11), (13, 17)]),
        Some(header + 3 * binding_header + 7 + 11 + 13 + 17)
    );
    assert_eq!(checked_retained_storage_bytes(usize::MAX, []), None);
    let largest_vector = usize::MAX / binding_header;
    assert_eq!(
        checked_retained_storage_bytes(largest_vector, []),
        header.checked_add(largest_vector * binding_header)
    );
    assert_eq!(checked_retained_storage_bytes(0, [(usize::MAX, 0)]), None);
    assert_eq!(checked_retained_storage_bytes(0, [(0, usize::MAX)]), None);
    let remaining = usize::MAX - header;
    assert_eq!(
        checked_retained_storage_bytes(0, [(remaining, 0)]),
        Some(usize::MAX)
    );
    assert_eq!(checked_retained_storage_bytes(0, [(remaining, 1)]), None);
    assert_eq!(
        checked_retained_storage_bytes(0, [(remaining, 0), (1, 0)]),
        None
    );
}
