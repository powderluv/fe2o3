#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_llvm;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

mod closure_profile_v1;
mod collector;
mod compiler_descriptor;
mod compiler_ffi_adapter;
mod compiler_module_contract;
mod device_ffi;
mod kernel_ir_codegen;
mod monomorphization_dead;
#[cfg(test)]
mod process_execution;
mod production_bf16_tile_values_source_v1;
mod production_canonical_phase_policy_v1;
mod production_complete_body_annotation_vnext;
mod production_complete_body_call_vnext;
mod production_complete_body_census_vnext;
mod production_complete_body_semantic_source_vnext;
mod production_complete_body_source_occurrences_vnext;
mod production_complete_body_terminal_v36;
mod production_conditional_reference_output_v1;
mod production_geometry_v1;
mod production_inline_assembly_v30;
mod production_inline_source_occurrences_v30;
mod production_mir_pliron_verus_join_v1;
mod production_native_source_lineage_v1;
mod production_ordered_composition_source_v1;
mod production_ordered_origin_report_v1;
mod production_ordered_program_source_occurrences_v32;
mod production_ordered_program_v32;
mod production_ordered_region_v31;
mod production_ordered_source_occurrences_v31;
mod production_physical_entry_annotation_v37;
mod production_physical_entry_call_v37;
mod production_physical_entry_census_v37;
mod production_physical_entry_source_abi_v37;
mod production_physical_entry_terminal_v37;
mod production_physical_global_copy_annotation_v38;
mod production_physical_global_copy_call_v38;
mod production_physical_global_copy_census_v38;
mod production_physical_global_copy_source_abi_v38;
mod production_physical_global_copy_terminal_v38;
mod production_physical_lds_exchange_annotation_v39;
mod production_physical_lds_exchange_call_v39;
mod production_physical_lds_exchange_census_v39;
mod production_physical_lds_exchange_source_abi_v39;
mod production_physical_lds_exchange_terminal_v39;
mod production_pipeline;
mod production_target_account;
mod production_tiled_region_source_v1;
pub use production_rustc_driver_v1::run_diagnostic_ordered_composition_extraction_driver_v1;
#[cfg(target_os = "linux")]
pub use production_rustc_driver_v1::run_ordered_composition_source_promotion_driver_v1;
mod production_policy;
mod production_primitive_from_v1;
mod production_ranked_projection_v1;
mod production_raw_call_audit_v1;
mod production_reference_bounds_v2;
mod production_reference_effect_join_v2;
#[cfg(test)]
mod production_rustc_driver_checked_output_source_helpers_v1_tests;
mod production_rustc_driver_v1;
#[cfg(target_os = "linux")]
mod source_bitselect_promotion_v1;
#[cfg(target_os = "linux")]
mod source_local_order_recipe_api_v1;
#[cfg(target_os = "linux")]
mod source_local_order_recipe_v1;
#[cfg(target_os = "linux")]
pub use production_rustc_driver_v1::run_bitselect_source_promotion_driver_v1;
#[cfg(target_os = "linux")]
pub use production_rustc_driver_v1::{
    run_source_local_order_recipe_driver_cancellable_v1,
    run_source_local_order_recipe_driver_measured_v1, run_source_local_order_recipe_driver_v1,
};
#[cfg(target_os = "linux")]
pub use source_bitselect_promotion_v1::{
    BitselectPromotionAttemptV1, BitselectPromotionFailureV1, BitselectPromotionRequestV1,
    CandidatePublicationStateV1, FailurePhaseV1, PublishedBitselectCandidateV1,
};
#[cfg(target_os = "linux")]
pub use source_local_order_recipe_api_v1::{
    SourceLocalOrderConstraintOutcomeV1, SourceLocalOrderIdentityObservationV1,
    SourceLocalOrderOrderV1, SourceLocalOrderRecipeAttemptV1,
    SourceLocalOrderRecipeCancellationCheckpointV1,
    SourceLocalOrderRecipeCancellationObservationV1, SourceLocalOrderRecipeCancellationRequestV1,
    SourceLocalOrderRecipeCancellationV1, SourceLocalOrderRecipeEvidenceV1,
    SourceLocalOrderRecipeFailurePhaseV1, SourceLocalOrderRecipeFailureV1,
    SourceLocalOrderRecipeOutputV1, SourceLocalOrderRecipeRequestV1,
    SourceLocalOrderRecipeRetainedStorageV1, SourceLocalOrderRelationV1,
    SourceLocalOrderSourceBindingModeV1, SourceLocalOrderStrengthV1,
};
mod production_core_panic_v50;
mod production_rustc_drop_v1;
mod production_rustc_intrinsic_v1;
mod production_rustc_slice_metadata_v1;
mod production_safe_core_shift_v1;
mod production_semantic_body_v1;
mod production_semantic_debug_v1;
mod production_semantic_fn_abi_v1;
mod production_semantic_lineage_v3;
mod production_semantic_terminal_v1;
mod production_semantic_types_v1;
mod production_target_v1;
mod production_worker_handoff;
mod protected_compiler_execution;
mod protected_rustc_invocation;
mod reference_effect_bijection_v1;
mod reference_effect_v1;
mod reference_enrollment_policy_v1;
mod rust_type_layout_general;
mod rust_type_layout_v3;
mod rustc_semantic_adapter_v1;
mod rustc_semantic_plan_v1;
pub mod semantic_layout_bridge;
mod static_registration;
#[cfg(test)]
mod test_temp_dir;
mod trusted_device_items;

/// Opaque move-only custody for an exact compiler-ranked root roster.
///
/// The compiler alone constructs and consumes this stage. It is named here so
/// compile-fail coverage can enforce that safe callers cannot clone it or
/// dismantle its private semantic and per-root custody.
///
/// ```compile_fail
/// use rustc_codegen_fe2o3::ProductionRankedSemanticProjectionRosterReceiptV1;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<ProductionRankedSemanticProjectionRosterReceiptV1>();
/// ```
///
/// ```compile_fail
/// use rustc_codegen_fe2o3::ProductionRankedSemanticProjectionRosterReceiptV1;
/// fn dismantle(receipt: ProductionRankedSemanticProjectionRosterReceiptV1) {
///     let ProductionRankedSemanticProjectionRosterReceiptV1 {
///         materialized: _,
///         source_order_roots: _,
///         canonical_kernel_order: _,
///         canonical_roster_identity: _,
///     } = receipt;
/// }
/// ```
#[doc(hidden)]
pub use production_ranked_projection_v1::ProductionRankedSemanticProjectionRosterReceiptV1;

#[doc(hidden)]
pub use production_rustc_driver_v1::{
    run_diagnostic_complete_body_kir_extraction_driver_v19,
    run_diagnostic_ordered_program_kir_extraction_driver_v17,
    run_diagnostic_ordered_program_origin_driver_v1,
    run_diagnostic_ordered_region_kir_extraction_driver_v16,
    run_diagnostic_physical_entry_extraction_driver_v20,
    run_diagnostic_physical_global_copy_extraction_driver_v21,
    run_diagnostic_physical_lds_exchange_extraction_driver_v22,
    run_production_amdgpu_compiler_handoff_capture_driver_v1,
    run_production_amdgpu_compiler_handoff_extraction_driver_v1,
    run_production_amdgpu_llvm_extraction_driver_v1, run_production_extraction_driver_v1,
    run_production_fixed_checked_output_extraction_driver_v1,
    run_production_fixed_checked_output_policy6_extraction_driver_v1,
    run_production_fixed_checked_output_policy7_extraction_driver_v1,
    run_production_fixed_checked_output_policy8_extraction_driver_v1,
    run_production_gfx942_compiler_handoff_extraction_driver_v1,
    run_production_gfx942_llvm_extraction_driver_v1, run_production_ranked_extraction_driver_v1,
    run_production_simulation_bundle_extraction_driver_v1,
    run_production_simulation_bundle_extraction_driver_v2,
    run_production_simulation_bundle_extraction_driver_v3,
    run_production_simulation_bundle_extraction_driver_v4,
    run_production_simulation_bundle_extraction_driver_v5,
    run_production_simulation_bundle_extraction_driver_v6,
};

use fe2o3_artifact_transaction as artifact_transaction;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModules, CrateInfo};
use rustc_data_structures::fx::FxIndexMap;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_session::Session;
use rustc_session::config::OutputFilenames;
use std::any::Any;
use std::env;
use std::fmt;
use std::path::{Path, PathBuf};

pub const TARGET_ENV: &str = "FE2O3_TARGET";
pub const BACKEND_ENV: &str = "FE2O3_BACKEND";
pub const VERBOSE_ENV: &str = "FE2O3_VERBOSE";
pub const DUMP_MIR_ENV: &str = "FE2O3_DUMP_MIR";
pub const DUMP_LLVM_ENV: &str = "FE2O3_DUMP_LLVM";
pub const OBSOLETE_CODEGEN_PIPELINE_ENV: &str = "FE2O3_CODEGEN_PIPELINE";
pub const HSACO_DIR_ENV: &str = "FE2O3_HSACO_DIR";
pub const BUILD_ATTEMPT_ENV: &str = artifact_transaction::BUILD_ATTEMPT_ENV_V1;

fn encode_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";

    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}

pub struct Fe2o3CodegenBackend {
    config: BackendConfig,
    llvm_backend: Box<dyn CodegenBackend>,
    compiler_execution_input: protected_compiler_execution::CompilerExecutionStartupInputV1,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
enum BuildAttemptSelection {
    #[default]
    Direct,
    Managed(artifact_transaction::BuildAttempt),
    Invalid(String),
}

impl BuildAttemptSelection {
    fn from_env() -> Self {
        match env::var(BUILD_ATTEMPT_ENV) {
            Err(env::VarError::NotPresent) => Self::Direct,
            Err(env::VarError::NotUnicode(_)) => {
                Self::Invalid(format!("{BUILD_ATTEMPT_ENV} is not valid UTF-8"))
            }
            Ok(value) => match artifact_transaction::BuildAttempt::from_env_value(&value) {
                Ok(attempt) => Self::Managed(attempt),
                Err(error) => Self::Invalid(format!(
                    "{BUILD_ATTEMPT_ENV} is not a canonical build attempt: {error}"
                )),
            },
        }
    }

    fn resolve(&self) -> Result<Option<artifact_transaction::BuildAttempt>, &str> {
        match self {
            Self::Direct => Ok(None),
            Self::Managed(attempt) => Ok(Some(*attempt)),
            Self::Invalid(reason) => Err(reason),
        }
    }
}
struct RetainedProductionDeviceAdmission<'b, 'w> {
    target: production_target_v1::RetainedProductionTargetV1,
    compiler_execution: protected_compiler_execution::dispatch::Admitted<'b, 'w>,
    build_attempt: artifact_transaction::BuildAttempt,
}

#[derive(Clone, Debug, Default)]
pub struct BackendConfig {
    pub verbose: bool,
    pub dump_mir: bool,
    pub dump_llvm: bool,
    production_environment_rejection: Option<String>,
    build_attempt: BuildAttemptSelection,
    pub hsaco_output_dir: Option<PathBuf>,
    pub target: AmdGpuTarget,
}

impl BackendConfig {
    pub fn from_env() -> Self {
        Self {
            verbose: env_flag(VERBOSE_ENV),
            dump_mir: env_flag(DUMP_MIR_ENV),
            dump_llvm: env_flag(DUMP_LLVM_ENV),
            production_environment_rejection: production_policy::environment_rejection(),
            build_attempt: BuildAttemptSelection::from_env(),
            hsaco_output_dir: env::var(HSACO_DIR_ENV).ok().map(PathBuf::from),
            target: AmdGpuTarget::from_env_or_default(),
        }
    }
}

fn has_custom_llvm_configuration(session: &Session) -> bool {
    !session.opts.cg.llvm_args.is_empty() || !session.opts.cg.passes.is_empty()
}

impl CodegenBackend for Fe2o3CodegenBackend {
    fn name(&self) -> &'static str {
        "fe2o3"
    }

    fn init(&self, sess: &Session) {
        self.llvm_backend.init(sess);
    }

    fn print_version(&self) {
        println!(
            "rustc-codegen-fe2o3 {} (wrapping rustc_codegen_llvm)",
            env!("CARGO_PKG_VERSION")
        );
        self.llvm_backend.print_version();
    }

    fn target_cpu(&self, sess: &Session) -> String {
        self.llvm_backend.target_cpu(sess)
    }

    fn target_config(&self, sess: &Session) -> rustc_codegen_ssa::TargetConfig {
        self.llvm_backend.target_config(sess)
    }

    fn provide(&self, providers: &mut rustc_middle::util::Providers) {
        self.llvm_backend.provide(providers);
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>, crate_info: &CrateInfo) -> Box<dyn Any> {
        with_no_trimmed_paths!({
            if let Some(reason) = &self.config.production_environment_rejection {
                tcx.dcx().fatal(format!(
                    "[rustc-codegen-fe2o3] production preflight failed: {reason}"
                ));
            }
            let mut protected_rustc_invocation =
                protected_rustc_invocation::admit_for_production_codegen().unwrap_or_else(|error| {
                    tcx.dcx().fatal(format!(
                        "[rustc-codegen-fe2o3] protected rustc invocation admission failed without fallback: {error}"
                    ))
                });
            let build_attempt = match self.config.build_attempt.resolve() {
                Ok(attempt) => attempt,
                Err(reason) => tcx.dcx().fatal(format!(
                    "[rustc-codegen-fe2o3] invalid managed build attempt: {reason}"
                )),
            };
            let production_root_count =
                collector::count_production_roots_before_monomorphization_v1(tcx);
            // The borrowed account spans admission, collection and publication.
            // No budget borrow or live service session escapes into LLVM codegen.
            production_target_account::with_device_phase(production_root_count > 0, |target_budget| {
                let mut production_device_admission = if production_root_count > 0 {
                    let build_attempt = build_attempt.unwrap_or_else(|| {
                        tcx.dcx().fatal(format!(
                            "[rustc-codegen-fe2o3] production compilation requires a managed {BUILD_ATTEMPT_ENV} before monomorphization"
                        ))
                    });
                    Some(RetainedProductionDeviceAdmission {
                        target: production_target_v1::RetainedProductionTargetV1::authenticate_before_collection(
                            tcx,
                            &self.config.target,
                        )
                        .unwrap_or_else(|error| {
                            tcx.dcx().fatal(format!(
                                "[rustc-codegen-fe2o3] production target authentication failed before monomorphization without fallback: {error}"
                                ))
                            }),
                        compiler_execution: self.compiler_execution_input.admit_selected(
                            target_budget.expect("device roots retain their original target account"),
                        )
                            .unwrap_or_else(|error| {
                                tcx.dcx().fatal(format!(
                                    "[rustc-codegen-fe2o3] protected compiler-execution admission failed without fallback: {error}"
                                ))
                            }),
                        build_attempt,
                    })
                } else {
                    None
                };
                let context_producers = collector::capture_context_producers_v1(tcx)
                    .unwrap_or_else(|error| {
                        tcx.dcx().fatal(format!(
                            "[rustc-codegen-fe2o3] context producer capture failed: {error}"
                        ))
                    });
                let mono_partitions = tcx.collect_and_partition_mono_items(());
                let kernel_count =
                    collector::count_kernels_in_cgus(tcx, mono_partitions.codegen_units);
                if production_device_admission.is_some() != (kernel_count > 0) {
                    let reason = if production_device_admission.is_some() {
                        "authenticated device roots disappeared during monomorphization"
                    } else {
                        "a device root appeared only after pre-monomorphization admission"
                    };
                    tcx.dcx().fatal(format!(
                        "[rustc-codegen-fe2o3] production root custody changed across monomorphization: {reason}; compilation failed closed"
                    ));
                }
                let crate_name = tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE);
                let output_dir = match managed_artifact_output(&self.config, kernel_count) {
                    Ok(output_dir) => output_dir,
                    Err(()) => tcx.dcx().fatal(format!(
                        "[rustc-codegen-fe2o3] {HSACO_DIR_ENV} must name a managed artifact directory when compiling kernels"
                    )),
                };
                if self.config.verbose || kernel_count > 0 {
                    eprintln!(
                        "[rustc-codegen-fe2o3] crate `{crate_name}`: {} CGU(s), {kernel_count} kernel candidate(s), target {}",
                        mono_partitions.codegen_units.len(),
                        self.config.target,
                    );
                }

                let mut production_device_transaction_complete = false;
                match production_pipeline::disposition(kernel_count) {
                    production_pipeline::ProductionDisposition::HostOnly => {}
                    production_pipeline::ProductionDisposition::DeviceTransaction => {
                        let RetainedProductionDeviceAdmission {
                            target,
                            compiler_execution,
                            build_attempt,
                        } = production_device_admission.take().expect(
                            "device admission presence was validated after monomorphization",
                        );
                        let has_custom_llvm_configuration = has_custom_llvm_configuration(tcx.sess);
                        if let Err(error) = production_pipeline::reject_custom_llvm_configuration(
                            has_custom_llvm_configuration,
                        ) {
                            tcx.dcx().fatal(format!("[rustc-codegen-fe2o3] {error}"));
                        }
                        let mut invocation = protected_rustc_invocation.take().unwrap_or_else(|| {
                            tcx.dcx().fatal(
                                "[rustc-codegen-fe2o3] production compilation requires protected rustc invocation custody",
                            )
                        });
                        let (compiler_execution, closure) = compiler_execution
                            .with_reference_enrollment(&mut invocation, |loan| {
                                Ok(match collector::collect_authenticated_kernel_closure_with_enrollment_v1(
                                tcx,
                                mono_partitions.codegen_units,
                                self.config.verbose,
                                target,
                                context_producers,
                                loan,
                            ) {
                                Ok(closure) => closure,
                                Err(error) => tcx.dcx().fatal(format!(
                                    "[rustc-codegen-fe2o3] production collection failed without fallback: {error}"
                                )),
                            })
                        }).unwrap_or_else(|error| tcx.dcx().fatal(format!(
                            "[rustc-codegen-fe2o3] reference enrollment failed without fallback: {error}"
                        )));
                        let output_dir = output_dir
                            .expect("device output was required above")
                            .to_path_buf();
                        let producer = match artifact_transaction::ProducerIdentity::from_rustc_invocation_descriptor_v3(
                            invocation.descriptor(),
                        ) {
                            Ok(producer) => producer,
                            Err(error) => tcx.dcx().fatal(format!(
                                "[rustc-codegen-fe2o3] protected rustc producer identity failed: {error}"
                            )),
                        };
                        let publication =
                            production_pipeline::ProductionCompilation::from_collected_device_closure(
                                tcx,
                                closure,
                                producer.clone(),
                                output_dir,
                                build_attempt,
                                invocation,
                            )
                            .and_then(|transaction| compiler_execution.publish(transaction));
                        match publication {
                            Ok(publication_length) => {
                                production_device_transaction_complete = true;
                                eprintln!(
                                    "[rustc-codegen-fe2o3] production compilation published {} canonical byte(s) of inert exact gfx942:xnack- LLVM handoff into the preselected managed compiler-module transaction; link, artifact, load, and launch authority remain false",
                                    publication_length,
                                );
                            }
                            Err(error) => tcx.dcx().fatal(format!("[rustc-codegen-fe2o3] {error}")),
                        }
                    }
                }
                if kernel_count > 0 && !production_device_transaction_complete {
                    tcx.dcx().fatal(
                        "[rustc-codegen-fe2o3] production compilation did not complete its device transaction; qualification fallback is forbidden",
                    );
                }
            }).unwrap_or_else(|error| {
                tcx.dcx().fatal(format!(
                    "[rustc-codegen-fe2o3] target account admission failed: {error}"
                ))
            });
            self.llvm_backend.codegen_crate(tcx, crate_info)
        })
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        self.llvm_backend
            .join_codegen(ongoing_codegen, sess, outputs)
    }

    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        self.llvm_backend
            .link(sess, compiled_modules, crate_info, metadata, outputs);
    }
}

/// Constructs the backend without consuming the dynamic loader's raw FD slots.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    let compiler_execution_input =
        protected_compiler_execution::CompilerExecutionStartupInputV1::capture();
    let config = BackendConfig::from_env();
    let llvm_backend = rustc_codegen_llvm::LlvmCodegenBackend::new();

    Box::new(Fe2o3CodegenBackend {
        config,
        llvm_backend,
        compiler_execution_input,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmdGpuTarget {
    name: String,
}

impl AmdGpuTarget {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn from_env_or_default() -> Self {
        env::var(TARGET_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(Self::new)
            .unwrap_or_else(|| Self::new("gfx1100"))
    }

    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl Default for AmdGpuTarget {
    fn default() -> Self {
        Self::from_env_or_default()
    }
}

impl fmt::Display for AmdGpuTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn managed_artifact_output(
    config: &BackendConfig,
    kernel_count: usize,
) -> Result<Option<&Path>, ()> {
    match config.hsaco_output_dir.as_deref() {
        Some(path) if !path.as_os_str().is_empty() => Ok(Some(path)),
        Some(_) => Err(()),
        None if kernel_count == 0 => Ok(None),
        None => Err(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendConfig, BuildAttemptSelection, managed_artifact_output};
    use std::path::{Path, PathBuf};
    #[test]
    fn admitted_protected_modules_publish_only_through_strict_v3() {
        let backend = include_str!("lib.rs");
        let production_pipeline = include_str!("production_pipeline.rs");
        let device_phase = backend
            .split("production_target_account::with_device_phase(production_root_count > 0,")
            .nth(1)
            .expect("one synchronous device phase")
            .split("self.llvm_backend.codegen_crate(tcx, crate_info)")
            .next()
            .expect("bounded device phase");
        let production = device_phase
            .split("let mut production_device_transaction_complete")
            .nth(1)
            .expect("production transaction tracking exists")
            .split("self.llvm_backend.codegen_crate(tcx, crate_info)")
            .next()
            .expect("bounded production transaction");
        assert!(production.contains("protected_rustc_invocation.take()"));
        assert!(backend.contains("self.compiler_execution_input.admit_selected("));
        assert!(production.contains("from_rustc_invocation_descriptor_v3"));
        assert!(production.contains("invocation.descriptor()"));
        assert!(!production.contains("local_crate_source_file"));
        assert!(!production.contains("ProducerIdentity::from_codegen"));
        assert!(production.contains("production_device_admission"));
        assert!(production.contains(".take()"));
        assert!(!production.contains("build_attempt.unwrap_or_else"));
        assert!(production.contains("from_collected_device_closure("));
        assert!(production.contains("compiler_execution.publish(transaction)"));
        assert!(!device_phase.contains(".with_budget("));
        assert!(!production.contains("from_collected_device_closure_with_protected_invocation_v3"));
        assert!(!production.contains("publish_worker_handoff_v3"));
        assert!(!production.contains("None =>"));
        assert!(!production.contains("QualificationOracle"));
        assert!(!production.contains("qualification_selection"));
        assert!(!production_pipeline.contains("Option<BuildAttempt>"));
        assert!(production_pipeline.contains("publish_compiler_module_handoff_v3"));
        assert!(production_pipeline.contains("publish_compiler_execution_receipt_transport_v1"));
        let admission = backend
            .find("let mut production_device_admission")
            .expect("production device admission");
        let monomorphization = backend
            .find("let mono_partitions = tcx.collect_and_partition_mono_items")
            .expect("monomorphization boundary");
        assert!(admission < monomorphization);
        let account = backend
            .find("production_target_account::with_device_phase(")
            .unwrap();
        let execution = backend
            .find("compiler_execution: self.compiler_execution_input.admit_selected(")
            .unwrap();
        assert!(account < admission && admission < execution && execution < monomorphization);
        let host_codegen = backend
            .find("self.llvm_backend.codegen_crate(tcx, crate_info)")
            .unwrap();
        assert!(account < host_codegen);
        assert!(!device_phase.contains("self.llvm_backend.codegen_crate"));
        let account_scope = include_str!("production_target_account.rs");
        assert!(account_scope.contains("run: impl FnOnce(Option<&mut Budget<'_>>) -> T"));
        assert!(account_scope.contains("account.with_budget(|budget| run(Some(budget)))"));
    }

    #[test]
    fn backend_configuration_is_feature_invariant_and_selector_free() {
        let backend = include_str!("lib.rs");
        let configuration = backend
            .split("pub struct BackendConfig")
            .nth(1)
            .expect("backend configuration exists")
            .split("impl CodegenBackend")
            .next()
            .expect("bounded backend configuration");
        assert!(configuration.contains("production_environment_rejection"));
        assert!(configuration.contains("production_policy::environment_rejection()"));
        assert!(!configuration.contains("QualificationSelection"));
        assert!(!configuration.contains("qualification_selection"));
    }

    #[test]
    fn kernels_require_an_explicit_managed_artifact_directory() {
        let mut config = BackendConfig::default();
        assert_eq!(managed_artifact_output(&config, 0), Ok(None));
        assert_eq!(managed_artifact_output(&config, 1), Err(()));

        config.hsaco_output_dir = Some(PathBuf::new());
        assert_eq!(managed_artifact_output(&config, 1), Err(()));

        config.hsaco_output_dir = Some(PathBuf::from("target/fe2o3"));
        assert_eq!(
            managed_artifact_output(&config, 1),
            Ok(Some(Path::new("target/fe2o3")))
        );
    }

    #[test]
    fn build_attempt_selection_is_direct_or_exactly_canonical() {
        assert_eq!(BuildAttemptSelection::default().resolve(), Ok(None));
        let value = concat!(
            "7:11111111111111111111111111111111:",
            "2222222222222222222222222222222222222222222222222222222222222222"
        );
        let selection = BuildAttemptSelection::Managed(
            fe2o3_artifact_transaction::BuildAttempt::from_env_value(value).unwrap(),
        );
        assert_eq!(selection.resolve().unwrap().unwrap().to_env_value(), value);
        assert!(
            BuildAttemptSelection::Invalid("bad".to_owned())
                .resolve()
                .is_err()
        );
    }
}

#[cfg(target_os = "linux")]
pub use production_rustc_driver_v1::{
    run_bf16_tile_source_inspection_driver_v1, run_bf16_tile_source_promotion_driver_v1,
};

#[cfg(target_os = "linux")]
pub use production_rustc_driver_v1::run_bf16_generated_source_admission_driver_v1;
