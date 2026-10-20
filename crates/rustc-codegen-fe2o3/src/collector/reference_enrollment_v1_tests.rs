//! Actual-rustc selector resolution and cumulative SOURCE work checks.
//! These tests construct no issuer, enrollment stamp, or admission authority.

use super::*;
use crate::test_temp_dir::TestTempDir;
use fe2o3_mir_model::semantic_mir_v1::{SemanticMirLimitsV1, SemanticMirResourceV1};
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;

const CRATE: &str = "fe2o3_reference_enrollment_fixture";
const SOURCE: &str = r#"
#![allow(dead_code)]
pub fn kernel(seed: u32) { let _value = seed ^ 3; }
pub fn reference(seed: u32) { let _value = seed ^ 3; }
pub fn renamed_kernel(seed: u32) { let _value = seed & 7; }
pub fn renamed_reference(seed: u32) { let _value = seed & 7; }
pub mod nested {
    pub mod deeper {
        pub fn nested_reference(seed: u32) { let _value = seed + 1; }
    }
}
pub mod associated {
    pub struct Host;
    impl Host {
        pub fn associated_reference(seed: u32) { let _value = seed ^ 9; }
    }
}
pub fn named_disambiguators() {
    {
        fn repeated_reference(seed: u32) { let _value = seed ^ 1; }
        let _first: fn(u32) = repeated_reference;
    }
    {
        fn repeated_reference(seed: u32) { let _value = seed ^ 2; }
        let _second: fn(u32) = repeated_reference;
    }
}
#[inline(never)]
pub fn generic_unique<T: Copy>(value: T) -> T { value }
#[inline(never)]
pub fn generic_ambiguous<T: Copy>(value: T) -> T { value }
#[inline(never)]
pub fn generic_absent<T: Copy>(value: T) -> T { value }
pub fn mono_anchor(seed: u32) -> u64 {
    generic_unique::<u32>(seed) as u64
        + generic_ambiguous::<u32>(seed) as u64
        + generic_ambiguous::<u64>(seed as u64)
}
"#;

struct CheckCallbacks(for<'tcx> fn(TyCtxt<'tcx>), bool);

impl Callbacks for CheckCallbacks {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        (self.0)(tcx);
        self.1 = true;
        Compilation::Stop
    }
}

fn with_source(check: for<'tcx> fn(TyCtxt<'tcx>)) {
    let directory = TestTempDir::create("fe2o3-reference-enrollment");
    let source = directory.path().join("fixture.rs");
    std::fs::write(&source, SOURCE).unwrap();
    let output = std::process::Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let sysroot = String::from_utf8(output.stdout).unwrap();
    let args = vec![
        "rustc".into(),
        format!("--crate-name={CRATE}"),
        "--crate-type=lib".into(),
        "--edition=2024".into(),
        // Metadata-only compilation omits reachable concrete CGU instances.
        // The after-analysis callback still stops before object emission.
        "--emit=obj".into(),
        "-Zmir-opt-level=0".into(),
        "-Copt-level=0".into(),
        "-Ccodegen-units=2".into(),
        "-Coverflow-checks=off".into(),
        "-Cpanic=abort".into(),
        "--sysroot".into(),
        sysroot.trim().into(),
        "-o".into(),
        directory.path().join("fixture.o").display().to_string(),
        source.display().to_string(),
    ];
    let mut callbacks = CheckCallbacks(check, false);
    rustc_driver::run_compiler(&args, &mut callbacks);
    assert!(callbacks.1, "source enrollment callback did not run");
    assert!(!directory.path().join("fixture.o").exists());
}

fn definition(tcx: TyCtxt<'_>, name: &str) -> rustc_hir::def_id::DefId {
    let mut definitions = tcx.hir_body_owners().filter(|id| {
        matches!(tcx.def_kind(id.to_def_id()), DefKind::Fn | DefKind::AssocFn)
            && tcx.item_name(id.to_def_id()).as_str() == name
    });
    let result = definitions
        .next()
        .unwrap_or_else(|| panic!("missing fixture function {name}"))
        .to_def_id();
    assert!(
        definitions.next().is_none(),
        "ambiguous fixture name {name}"
    );
    result
}

fn assert_rejected<T>(result: Result<T, ReferenceBindingErrorV1>, reason: &str) {
    let Err(error) = result else {
        panic!("expected rejection containing {reason:?}");
    };
    assert!(error.to_string().contains(reason), "{error}");
}

fn check_work_boundaries<T>(
    operation: impl Fn(&mut SourceClosureWorkV1) -> Result<T, ReferenceBindingErrorV1>,
) {
    let mut measured = SourceClosureWorkV1::default();
    operation(&mut measured).unwrap_or_else(|error| panic!("measuring source work: {error}"));
    let cost = measured.validation_work_for_test();
    assert!(cost > 0);
    let limit = SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::ValidationWork);
    for inherited in [0, 17] {
        let mut work = SourceClosureWorkV1::default();
        work.charge(inherited).unwrap();
        operation(&mut work).unwrap_or_else(|error| panic!("inherited source work: {error}"));
        assert_eq!(work.validation_work_for_test(), inherited as u64 + cost);
    }
    for remaining in [cost, cost - 1, 0] {
        let mut work = SourceClosureWorkV1::default();
        work.charge(17).unwrap();
        work.charge(usize::try_from(limit - work.validation_work_for_test() - remaining).unwrap())
            .unwrap();
        let result = operation(&mut work);
        if remaining == cost {
            result.unwrap_or_else(|error| panic!("exact source work: {error}"));
            assert_eq!(work.validation_work_for_test(), limit);
        } else {
            assert_rejected(result, "ValidationWork");
            let first = work.validation_work_for_test();
            assert!(first > limit);
            assert_rejected(operation(&mut work), "ValidationWork");
            assert!(work.validation_work_for_test() > first);
        }
    }
}

#[test]
fn source_selector_matches_exact_crate_module_and_renamed_functions() {
    with_source(|tcx| {
        for (name, selector) in [
            ("kernel", "fe2o3_reference_enrollment_fixture::kernel"),
            ("reference", "fe2o3_reference_enrollment_fixture::reference"),
            (
                "renamed_kernel",
                "fe2o3_reference_enrollment_fixture::renamed_kernel",
            ),
            (
                "renamed_reference",
                "fe2o3_reference_enrollment_fixture::renamed_reference",
            ),
            (
                "nested_reference",
                "fe2o3_reference_enrollment_fixture::nested::deeper::nested_reference",
            ),
        ] {
            assert!(
                matches_selector(
                    tcx,
                    definition(tcx, name),
                    selector,
                    &mut SourceClosureWorkV1::default(),
                )
                .unwrap(),
                "{selector}"
            );
        }
    });
}

#[test]
fn source_selector_rejects_partial_wrong_and_noncanonical_qualification() {
    with_source(|tcx| {
        let id = definition(tcx, "nested_reference");
        for selector in [
            "",
            "nested_reference",
            "nested::deeper::nested_reference",
            "other_crate::nested::deeper::nested_reference",
            "fe2o3_reference_enrollment_fixture::deeper::nested_reference",
            "fe2o3_reference_enrollment_fixture::nested::wrong::nested_reference",
            "fe2o3_reference_enrollment_fixture::nested::deeper::reference",
            "::fe2o3_reference_enrollment_fixture::nested::deeper::nested_reference",
            "fe2o3_reference_enrollment_fixture::nested::deeper::nested_reference::",
            "fe2o3_reference_enrollment_fixture::nested::deeper::nested_reference#0",
        ] {
            assert!(
                !matches_selector(tcx, id, selector, &mut SourceClosureWorkV1::default()).unwrap(),
                "{selector}"
            );
        }
    });
}

#[test]
fn source_selector_uses_canonical_anonymous_impl_disambiguators() {
    with_source(|tcx| {
        let id = definition(tcx, "associated_reference");
        let canonical =
            "fe2o3_reference_enrollment_fixture::associated::{impl#0}::associated_reference";
        assert!(matches_selector(tcx, id, canonical, &mut SourceClosureWorkV1::default()).unwrap());
        for selector in [
            "fe2o3_reference_enrollment_fixture::associated::Host::associated_reference",
            "fe2o3_reference_enrollment_fixture::associated::impl#0::associated_reference",
            "fe2o3_reference_enrollment_fixture::associated::{impl}::associated_reference",
            "fe2o3_reference_enrollment_fixture::associated::{impl#00}::associated_reference",
            "fe2o3_reference_enrollment_fixture::associated::{impl#1}::associated_reference",
            "fe2o3_reference_enrollment_fixture::associated::{impl#+0}::associated_reference",
            "fe2o3_reference_enrollment_fixture::associated::{impl#4294967296}::associated_reference",
        ] {
            assert!(
                !matches_selector(tcx, id, selector, &mut SourceClosureWorkV1::default()).unwrap(),
                "{selector}"
            );
        }
    });
}

#[test]
fn source_selector_distinguishes_actual_named_definition_disambiguators() {
    with_source(|tcx| {
        let mut definitions = tcx
            .hir_body_owners()
            .filter(|id| {
                tcx.def_kind(id.to_def_id()) == DefKind::Fn
                    && tcx.item_name(id.to_def_id()).as_str() == "repeated_reference"
            })
            .map(|id| {
                let id = id.to_def_id();
                (tcx.def_key(id).disambiguated_data.disambiguator, id)
            })
            .collect::<Vec<_>>();
        definitions.sort_by_key(|(ordinal, _)| *ordinal);
        assert_eq!(
            definitions
                .iter()
                .map(|(ordinal, _)| *ordinal)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        let selectors = [
            "fe2o3_reference_enrollment_fixture::named_disambiguators::repeated_reference",
            "fe2o3_reference_enrollment_fixture::named_disambiguators::repeated_reference#1",
        ];
        for (ordinal, (_, id)) in definitions.iter().enumerate() {
            for (selected, selector) in selectors.iter().enumerate() {
                assert_eq!(
                    matches_selector(tcx, *id, selector, &mut SourceClosureWorkV1::default())
                        .unwrap(),
                    ordinal == selected,
                    "{selector}",
                );
            }
        }
        for selector in [
            "fe2o3_reference_enrollment_fixture::named_disambiguators::repeated_reference#0",
            "fe2o3_reference_enrollment_fixture::named_disambiguators::repeated_reference#01",
            "fe2o3_reference_enrollment_fixture::named_disambiguators::repeated_reference#2",
            "fe2o3_reference_enrollment_fixture::named_disambiguators::repeated_reference#4294967296",
        ] {
            for (_, id) in &definitions {
                assert!(
                    !matches_selector(tcx, *id, selector, &mut SourceClosureWorkV1::default())
                        .unwrap()
                );
            }
        }
    });
}

#[test]
fn source_reference_resolver_returns_actual_unique_nongeneric_instances() {
    with_source(|tcx| {
        for (name, selector) in [
            ("reference", "fe2o3_reference_enrollment_fixture::reference"),
            (
                "renamed_reference",
                "fe2o3_reference_enrollment_fixture::renamed_reference",
            ),
            (
                "nested_reference",
                "fe2o3_reference_enrollment_fixture::nested::deeper::nested_reference",
            ),
            (
                "associated_reference",
                "fe2o3_reference_enrollment_fixture::associated::{impl#0}::associated_reference",
            ),
        ] {
            let actual =
                reference_instance_v1(tcx, &[], selector, &mut SourceClosureWorkV1::default())
                    .unwrap();
            assert_eq!(actual, Instance::mono(tcx, definition(tcx, name)));
            assert!(actual.def_id().is_local());
            assert!(is_fully_monomorphized(tcx, actual));
        }
    });
}

#[test]
fn source_reference_resolver_rejects_nonlocal_unresolved_and_nonfunction_selectors() {
    with_source(|tcx| {
        for selector in [
            "core::mem::drop",
            "std::mem::drop",
            "fe2o3_reference_enrollment_fixture::missing_reference",
            "other_crate::reference",
            "reference",
            "fe2o3_reference_enrollment_fixture::associated::Host",
            "fe2o3_reference_enrollment_fixture::nested",
        ] {
            assert_rejected(
                reference_instance_v1(tcx, &[], selector, &mut SourceClosureWorkV1::default()),
                "not a local function",
            );
        }
    });
}

#[test]
fn source_reference_resolver_does_not_invent_generic_arguments_without_cgus() {
    with_source(|tcx| {
        for selector in [
            "fe2o3_reference_enrollment_fixture::generic_unique",
            "fe2o3_reference_enrollment_fixture::generic_absent",
        ] {
            assert_rejected(
                reference_instance_v1(tcx, &[], selector, &mut SourceClosureWorkV1::default()),
                "no actual monomorphized instance",
            );
        }
    });
}

#[test]
fn source_reference_resolver_accepts_one_actual_generic_monomorphization() {
    with_source(|tcx| {
        let partitions = tcx.collect_and_partition_mono_items(());
        let actual = reference_instance_v1(
            tcx,
            partitions.codegen_units,
            "fe2o3_reference_enrollment_fixture::generic_unique",
            &mut SourceClosureWorkV1::default(),
        )
        .unwrap();
        assert_eq!(actual.def_id(), definition(tcx, "generic_unique"));
        assert_eq!(actual.args.type_at(0), tcx.types.u32);
        assert!(is_fully_monomorphized(tcx, actual));
        assert!(partitions.codegen_units.iter().any(|cgu| {
            cgu.items()
                .keys()
                .any(|item| matches!(item, MonoItem::Fn(instance) if *instance == actual))
        }));
    });
}

#[test]
fn source_reference_resolver_rejects_distinct_actual_generic_monomorphizations() {
    with_source(|tcx| {
        let partitions = tcx.collect_and_partition_mono_items(());
        let target = definition(tcx, "generic_ambiguous");
        let mut actual = Vec::new();
        for cgu in partitions.codegen_units {
            for item in cgu.items().keys() {
                if let MonoItem::Fn(instance) = item {
                    if instance.def_id() == target && !actual.contains(instance) {
                        actual.push(*instance);
                    }
                }
            }
        }
        assert_eq!(
            actual.len(),
            2,
            "fixture must contain two actual monomorphizations"
        );
        assert_rejected(
            reference_instance_v1(
                tcx,
                partitions.codegen_units,
                "fe2o3_reference_enrollment_fixture::generic_ambiguous",
                &mut SourceClosureWorkV1::default(),
            ),
            "ambiguous monomorphizations",
        );
    });
}

#[test]
fn source_reference_resolver_rejects_generic_items_absent_from_actual_cgus() {
    with_source(|tcx| {
        let partitions = tcx.collect_and_partition_mono_items(());
        assert!(!partitions.codegen_units.is_empty());
        assert_rejected(
            reference_instance_v1(
                tcx,
                partitions.codegen_units,
                "fe2o3_reference_enrollment_fixture::generic_absent",
                &mut SourceClosureWorkV1::default(),
            ),
            "no actual monomorphized instance",
        );
    });
}

#[test]
fn source_selector_work_is_inherited_exact_bounded_and_sticky() {
    with_source(|tcx| {
        for (name, selector) in [
            (
                "nested_reference",
                "fe2o3_reference_enrollment_fixture::nested::deeper::nested_reference",
            ),
            (
                "associated_reference",
                "fe2o3_reference_enrollment_fixture::associated::{impl#0}::associated_reference",
            ),
        ] {
            let id = definition(tcx, name);
            check_work_boundaries(|work| {
                let matched = matches_selector(tcx, id, selector, work)?;
                assert!(matched);
                Ok(())
            });
        }
    });
}

#[test]
fn source_reference_resolution_shares_exact_cumulative_work() {
    with_source(|tcx| {
        check_work_boundaries(|work| {
            reference_instance_v1(
                tcx,
                &[],
                "fe2o3_reference_enrollment_fixture::renamed_reference",
                work,
            )
        });
        let partitions = tcx.collect_and_partition_mono_items(());
        check_work_boundaries(|work| {
            reference_instance_v1(
                tcx,
                partitions.codegen_units,
                "fe2o3_reference_enrollment_fixture::generic_unique",
                work,
            )
        });
    });
}

#[test]
fn source_selector_misses_do_not_reset_work_or_accept_an_exhausted_account() {
    with_source(|tcx| {
        let id = definition(tcx, "reference");
        let mut work = SourceClosureWorkV1::default();
        work.charge(17).unwrap();
        let selector = "other_crate::reference";
        assert!(!matches_selector(tcx, id, selector, &mut work).unwrap());
        let first = work.validation_work_for_test();
        assert!(first > 17);
        assert!(!matches_selector(tcx, id, selector, &mut work).unwrap());
        assert_eq!(work.validation_work_for_test(), 17 + 2 * (first - 17));
        let limit = SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::ValidationWork);
        work.charge(usize::try_from(limit - work.validation_work_for_test()).unwrap())
            .unwrap();
        assert_rejected(
            matches_selector(tcx, id, selector, &mut work),
            "ValidationWork",
        );
        let refused = work.validation_work_for_test();
        assert_rejected(
            matches_selector(tcx, id, selector, &mut work),
            "ValidationWork",
        );
        assert!(work.validation_work_for_test() > refused);
    });
}
