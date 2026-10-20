//! Actual empty-roster bind/capture/rederive under an admitted issuer loan.
//!
//! The rustc callback supplies a real Session. There are no registered device
//! roots, so this deliberately does not construct an authenticated production
//! closure, run the production importer, freeze a recipe or publish a receipt.
//! Those complete protected-compiler integration tests remain required.

use super::{reference_custody_v1::RetainedReferenceInputsV1, reference_enrollment_v1};
use crate::protected_compiler_execution::native_v3::{Admitted, Error};
use crate::protected_rustc_invocation::AdmittedProtectedRustcInvocationV1 as Invocation;
use crate::rustc_semantic_plan_v1::SourceClosureWorkV1 as Work;
use crate::test_temp_dir::TestTempDir;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;

struct CheckCallbacks<F> {
    check: F,
    calls: usize,
}

impl<F: for<'tcx> FnMut(TyCtxt<'tcx>)> Callbacks for CheckCallbacks<F> {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        self.calls += 1;
        (self.check)(tcx);
        Compilation::Stop
    }
}

fn linked_driver_sysroot() -> std::path::PathBuf {
    // The isolated child intentionally has no PATH or rustup selection. Resolve
    // the loaded rustc_driver in process, never a cwd-dependent rustc proxy.
    for key in ["PATH", "HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN"] {
        assert!(std::env::var_os(key).is_none(), "unexpected {key}");
    }
    let argv0 = std::path::PathBuf::from(std::env::args_os().next().unwrap());
    assert!(argv0.is_absolute());
    assert_eq!(argv0, argv0.canonicalize().unwrap());
    assert!(std::fs::symlink_metadata(&argv0).unwrap().is_file());
    // The pinned Sysroot implementation tries an argv[0] symlink before the
    // loaded driver. Reject that alternative above, even if it has a rustlib.
    let sysroot = rustc_session::config::Sysroot::new(None);
    let sysroot = sysroot.path().canonicalize().unwrap();
    assert!(
        sysroot
            .join("lib/rustlib")
            .join(rustc_session::config::host_tuple())
            .join("lib")
            .is_dir()
    );
    sysroot
}

// A capturing callback keeps the original move-only native session outside it.
// This fixed source has no proc macros; Stop after analysis avoids codegen and
// linking. No helper process is created beneath the deadline-owned child.
fn with_source(check: impl for<'tcx> FnMut(TyCtxt<'tcx>) + Send) {
    let directory = TestTempDir::create("fe2o3-loan-empty-roster");
    let source = directory.path().join("fixture.rs");
    std::fs::write(&source, "pub fn unregistered(value: u32) -> u32 { value }").unwrap();
    let sysroot = linked_driver_sysroot();
    let args = vec![
        "rustc".into(),
        "--crate-name=fe2o3_loan_empty_roster".into(),
        "--crate-type=lib".into(),
        "--edition=2024".into(),
        "--emit=metadata".into(),
        "-Zmir-opt-level=0".into(),
        "-Copt-level=0".into(),
        "-Cpanic=abort".into(),
        "--sysroot".into(),
        sysroot.to_str().unwrap().into(),
        "-o".into(),
        directory.path().join("fixture.rmeta").display().to_string(),
        source.display().to_string(),
    ];
    let mut callbacks = CheckCallbacks { check, calls: 0 };
    rustc_driver::run_compiler(&args, &mut callbacks);
    assert_eq!(callbacks.calls, 1);
}

pub(crate) fn check_empty_enrollment_replay(native: Admitted<'_, '_>, invocation: &mut Invocation) {
    let mut native = Some(native);
    with_source(|tcx| {
        let mut source = Work::default();
        source.charge(17).unwrap();
        let (session, (retained, owner_check_cost, request_cost)) = native
            .take()
            .unwrap()
            .with_reference_enrollment::<_, Error>(invocation, |loan| {
                let loan = loan.unwrap();
                // Measure the whole original request boundary, including count
                // projection and custody checks, on an independent source meter.
                let mut request_meter = Work::default();
                assert!(loan.request(&mut request_meter)?.is_none());
                let request_cost = request_meter.validation_work_for_test();
                let enrollment =
                    reference_enrollment_v1::bind_v1(tcx, &[], &mut [], loan, &mut source)?;
                let before = source.validation_work_for_test();
                enrollment.revalidate(tcx, loan, &mut source)?;
                let owner_check_cost = source.validation_work_for_test() - before;
                let retained = RetainedReferenceInputsV1::capture_with_enrollment(
                    tcx,
                    &[],
                    &mut source,
                    Some(enrollment),
                    Some(loan),
                )?;
                Ok((retained, owner_check_cost, request_cost))
            })
            .unwrap();
        // Moving the original SOURCE account is permitted; constructing an
        // equal counter below must not supply its retained nominal identity.
        let mut source = source;
        let mut session = session;
        for _ in 0..2 {
            let (next, ()) = session
                .with_reference_enrollment::<_, Error>(invocation, |loan| {
                    let loan = loan.unwrap();
                    let error = retained.rederive(tcx, &[], &mut source).unwrap_err();
                    assert_eq!(
                        error.to_string(),
                        "reference enrollment live owner presence changed"
                    );
                    let before = source.validation_work_for_test();
                    // Do not request here: replay itself must request on this fresh
                    // loan, including the no-mapping case. Count its exact debit.
                    let bindings =
                        retained.rederive_with_enrollment(tcx, &[], &mut source, Some(loan))?;
                    assert!(bindings.as_slice().is_empty());
                    assert_eq!(
                        source.validation_work_for_test() - before,
                        owner_check_cost + request_cost + 1
                    );
                    Ok(())
                })
                .unwrap();
            session = next;
        }
        let mut foreign = Work::default();
        let _foreign_stamp = foreign.retain_identity_v1().unwrap();
        foreign
            .charge(
                usize::try_from(
                    source.validation_work_for_test() - foreign.validation_work_for_test(),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            foreign.validation_work_for_test(),
            source.validation_work_for_test()
        );
        let result = session.with_reference_enrollment::<_, Error>(invocation, |loan| {
            let error = retained
                .rederive_with_enrollment(tcx, &[], &mut foreign, loan)
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "reference enrollment source account changed"
            );
            Err::<(), _>(Error::from(error))
        });
        let Err(error) = result else {
            panic!("foreign SOURCE account accepted")
        };
        assert_eq!(
            error.to_string(),
            "reference enrollment source account changed"
        );
    });
    assert!(native.is_none());
}
