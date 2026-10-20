use super::*;
use std::ffi::OsString;

const REQUEST: &str =
    r#"{"version":1,"bindings":[{"kernel":"demo::write","reference":"demo::spec"}]}"#;

fn decode(text: &str) -> Result<ReferenceEnrollmentRequestV1, Error> {
    ReferenceEnrollmentRequestV1::decode(text, &mut SourceClosureWorkV1::default())
}

#[test]
fn explicit_selection_is_generic_and_does_not_require_matching_function_names() {
    let request = decode(REQUEST).unwrap();
    assert_eq!(request.bindings()[0].kernel(), "demo::write");
    assert_eq!(request.bindings()[0].reference(), "demo::spec");
    let renamed = REQUEST.replace("demo::write", "other::renamed");
    assert_eq!(
        decode(&renamed).unwrap().bindings()[0].kernel(),
        "other::renamed"
    );
}

#[test]
fn multiple_kernels_may_select_the_same_reference() {
    let request = decode(
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"spec"},{"kernel":"b","reference":"spec"}]}"#,
    ).unwrap();
    assert_eq!(request.bindings().len(), 2);
}

#[test]
fn schema_rejects_unknown_duplicate_missing_fields_and_trailing_values() {
    for text in [
        r#"[1,[{"kernel":"a","reference":"b"}]]"#,
        r#"{"version":1,"bindings":[["a","b"]]}"#,
        r#"{"version":1,"version":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[],"authority":true}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","kernel":"a","reference":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"b","proof":true}]}"#,
        r#"{"bindings":[{"kernel":"a","reference":"b"}]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a"}]}"#,
        r#"{"version":1,"bindings":null}"#,
        r#"{"version":1,"bindings":[]} null"#,
    ] {
        assert_eq!(
            decode(text).unwrap_err().to_string(),
            "invalid reference enrollment request schema",
            "{text}"
        );
    }
}

#[test]
fn version_empty_roster_duplicates_and_reordering_are_distinct_refusals() {
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
}

#[test]
fn selector_limits_apply_to_decoded_bytes_without_reimplementing_rust_resolution() {
    for value in [
        String::new(),
        "x".repeat(MAX_SELECTOR_BYTES + 1),
        "a\nb".to_owned(),
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
    json["bindings"][0]["kernel"] = "x".repeat(MAX_SELECTOR_BYTES).into();
    assert!(decode(&json.to_string()).is_ok());
    assert!(decode(r#"{"version":1,"bindings":[{"kernel":"a\u0062","reference":"r"}]}"#).is_ok());
}

#[test]
fn encoded_size_bounds_the_aggregate_roster_and_all_selector_bytes() {
    let padded = format!("{REQUEST}{}", " ".repeat(MAX_BYTES - REQUEST.len()));
    assert!(decode(&padded).is_ok());
    assert_eq!(
        decode(&(padded + " ")).unwrap_err().to_string(),
        "reference enrollment request exceeds byte limit"
    );
    let rows = |count| {
        serde_json::json!({"version":1,"bindings":
            (0..count).map(|i| serde_json::json!({
                "kernel":format!("k{i:04}"),"reference":"r"
            })).collect::<Vec<_>>()
        })
        .to_string()
    };
    assert!(decode(&rows(64)).is_ok());
    assert_eq!(
        decode(&rows(MAX_BINDINGS + 1)).unwrap_err().to_string(),
        "reference enrollment request exceeds byte limit"
    );
}

#[test]
fn reads_only_the_captured_environment_and_preserves_absence() {
    let environment = |entries: Vec<(&str, &str)>| {
        CompileEnvironmentV2::from_child_environment(
            entries
                .into_iter()
                .map(|(k, v)| (OsString::from(k), OsString::from(v))),
        )
        .unwrap()
    };
    let empty = environment(vec![("OTHER", REQUEST)]);
    assert!(
        ReferenceEnrollmentRequestV1::from_environment(&empty, &mut SourceClosureWorkV1::default())
            .unwrap()
            .is_none()
    );
    let present = environment(vec![(ENV, REQUEST)]);
    let mut work = SourceClosureWorkV1::default();
    work.charge(17).unwrap();
    let request = ReferenceEnrollmentRequestV1::from_environment(&present, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(request, decode(REQUEST).unwrap());
    let first = work.validation_work_for_test();
    assert!(first > 17);
    ReferenceEnrollmentRequestV1::from_environment(&present, &mut work).unwrap();
    assert_eq!(work.validation_work_for_test() - first, first - 17);
}
