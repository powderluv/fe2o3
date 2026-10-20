//! Inert calculation/accounting tests, not native Backing/Attempt admission.
use super::*;
use fe2o3_build_authority::CompilerClosureV2;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use fe2o3_rustc_invocation::{
    CompileEnvironmentV2, REFERENCE_ENROLLMENT_ENV_V1, RustcInvocationDescriptorV2, RustcUnitV2,
};

const PREFIX: usize = 31;
const ONE: &str = r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"}]}"#;
const TWO: &str =
    r#"{"version":1,"bindings":[{"kernel":"a","reference":"r"},{"kernel":"b","reference":"s"}]}"#;

fn capture(request: Option<&str>, source: u8) -> Capture {
    let mut environment = vec![
        ("FE2O3_TARGET".into(), "gfx942:xnack-".into()),
        ("FE2O3_HSACO_DIR".into(), "/proc/self/fd/197".into()),
    ];
    if let Some(request) = request {
        environment.push((REFERENCE_ENROLLMENT_ENV_V1.into(), request.into()));
    }
    let v2 = RustcInvocationDescriptorV2::new(
        [0x11; 32],
        [0x22; 32],
        RustcUnitV2::new(
            "/workspace/project",
            vec![
                "/toolchains/rustc".into(),
                "-Zcodegen-backend=/proc/./self/fd/198".into(),
            ],
        )
        .unwrap(),
        CompileEnvironmentV2::from_child_environment(environment).unwrap(),
    )
    .unwrap();
    let closure = CompilerClosureV2::new(
        [source; 32],
        [0x32; 32],
        [0x33; 32],
        [0x11; 32],
        [0x35; 32],
        [0x22; 32],
    )
    .unwrap();
    Capture::create(Descriptor::new(v2, closure).unwrap()).unwrap()
}

fn quote(capture: &Capture) -> usize {
    let mut work = ENTRY
        + 2 * capture.canonical_bytes().len()
        + INVOCATION_DIGEST_DOMAIN_V3.len()
        + size_of::<u64>()
        + 1;
    Request::project_binding_count_from_descriptor(capture.descriptor(), |n| {
        work = work.checked_add(n).unwrap();
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap();
    work
}

#[test]
fn original_sealed_preimage_supplies_count_and_digest_without_a_header() {
    for (request, count) in [(None, None), (Some(ONE), Some(1)), (Some(TWO), Some(2))] {
        let capture = capture(request, 0x31);
        let cost = quote(&capture);
        let floor = PREFIX + capture.native_retained_storage().unwrap();
        let mut work = Work::new(PREFIX + cost);
        let mut b = Budget::new(&mut work, floor + SCRATCH);
        b.charge_work(PREFIX).unwrap();
        b.reserve_storage(floor).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let address = &b as *const Budget<'_>;
        let result = project(&capture, &mut b).unwrap();
        assert_eq!(result.binding_count, count);
        let expected: [u8; 32] = Sha256::digest(capture.canonical_bytes()).into();
        assert_eq!(result.rustc_invocation_sha256, expected);
        assert_eq!(
            result.intake_invocation_identity,
            InvocationDigestV3::calculate(capture.descriptor())
                .unwrap()
                .into_bytes()
        );
        assert_ne!(result.intake_invocation_identity, expected);
        assert_eq!(
            result.invocation_bytes,
            capture.canonical_bytes().len() as u64
        );
        assert_eq!(
            (b.work(), b.storage(), b.peak_storage()),
            (PREFIX + cost, floor, floor + SCRATCH)
        );
        assert!(ledger == b.work_ledger_identity_v1());
        assert_eq!(address, &b as *const Budget<'_>);
    }
}

#[test]
fn projection_ceiling_covers_absence_and_maximum_encoded_request() {
    let padded = format!(
        "{ONE}{}",
        " ".repeat(fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BYTES_V1 - ONE.len())
    );
    for request in [None, Some(ONE), Some(padded.as_str())] {
        let capture = capture(request, 0x31);
        let cost = quote(&capture);
        assert!(cost <= OriginalInvocationEnrollment::PROJECTION_WORK);
        let floor = capture.native_retained_storage().unwrap();
        let mut work = Work::new(OriginalInvocationEnrollment::PROJECTION_WORK);
        let mut b = Budget::new(
            &mut work,
            floor + OriginalInvocationEnrollment::PROJECTION_SCRATCH,
        );
        b.reserve_storage(floor).unwrap();
        assert_eq!(
            project(&capture, &mut b).unwrap().binding_count,
            request.map(|_| 1)
        );
        assert_eq!(b.work(), cost);
        assert_eq!(b.storage(), floor);
    }
}

#[test]
fn equal_counts_do_not_hide_changed_original_source_or_reference() {
    let captures = [
        capture(Some(ONE), 0x31),
        capture(Some(ONE), 0x41),
        capture(
            Some(r#"{"version":1,"bindings":[{"kernel":"a","reference":"changed"}]}"#),
            0x31,
        ),
    ];
    let mut digests = Vec::new();
    for capture in &captures {
        let floor = capture.native_retained_storage().unwrap();
        let mut work = Work::new(quote(capture));
        let mut b = Budget::new(&mut work, floor + SCRATCH);
        b.reserve_storage(floor).unwrap();
        let result = project(capture, &mut b).unwrap();
        assert_eq!(result.binding_count, Some(1));
        assert!(!digests.contains(&result.rustc_invocation_sha256));
        digests.push(result.rustc_invocation_sha256);
    }
}

#[test]
fn missing_input_byte_short_scratch_and_short_work_refuse_without_refund() {
    let capture = capture(Some(ONE), 0x31);
    let input = capture.native_retained_storage().unwrap();
    let cost = quote(&capture);
    for case in 0..3 {
        let floor = input - usize::from(case == 0);
        let mut work = Work::new(cost - usize::from(case == 2));
        let mut b = Budget::new(&mut work, floor + SCRATCH - usize::from(case == 1));
        b.reserve_storage(floor).unwrap();
        assert!(matches!(
            project(&capture, &mut b),
            Err(CompilerInvocationBackingError::Resource(_))
        ));
        assert_eq!(b.storage(), floor);
        if case != 0 {
            let history = (b.work(), b.storage(), b.failed_work(), b.failed_storage());
            assert!(project(&capture, &mut b).is_err());
            assert_eq!(
                (b.work(), b.storage(), b.failed_work(), b.failed_storage()),
                history
            );
        }
    }
}

#[test]
fn malformed_request_never_becomes_absent_or_zero() {
    for request in [
        "{}",
        r#"{"version":1,"bindings":[]}"#,
        r#"{"version":1,"bindings":[{"kernel":"a","reference":"r","extra":0}]}"#,
    ] {
        let capture = capture(Some(request), 0x31);
        let floor = capture.native_retained_storage().unwrap();
        let mut work = Work::new(1 << 20);
        let mut b = Budget::new(&mut work, floor + SCRATCH);
        b.reserve_storage(floor).unwrap();
        assert!(matches!(
            project(&capture, &mut b),
            Err(CompilerInvocationBackingError::Enrollment(_))
        ));
        assert_eq!(b.storage(), floor);
    }
}

#[test]
fn prior_denials_refuse_before_even_the_input_is_accessed() {
    let capture = capture(Some(ONE), 0x31);
    for storage in [false, true] {
        let mut work = Work::new(100);
        let mut b = Budget::new(&mut work, 100);
        b.charge_work(PREFIX).unwrap();
        b.reserve_storage(PREFIX).unwrap();
        if storage {
            assert!(b.reserve_storage(101).is_err());
        } else {
            assert!(b.charge_work(101).is_err());
        }
        let history = (
            b.work(),
            b.storage(),
            b.peak_storage(),
            b.failed_work(),
            b.failed_storage(),
        );
        assert!(matches!(
            project(&capture, &mut b),
            Err(CompilerInvocationBackingError::Resource(_))
        ));
        assert_eq!(
            (
                b.work(),
                b.storage(),
                b.peak_storage(),
                b.failed_work(),
                b.failed_storage()
            ),
            history
        );
    }
}
