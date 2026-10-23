// This shard shares private correspondence and lowering helpers with its owner.

include!("production_pre_ranked_local_frames_v1.rs");
include!("production_local_helper_erased_owner_v1.rs");
include!("production_local_helper_ranked_stage_v1.rs");
include!("production_unit_local_source_replay_v1.rs");
include!("production_local_helper_deletion_v1.rs");

#[path = "production_bf16_ranked_module_receipt_v1.rs"]
mod bf16_ranked_module_receipt_v1;
#[doc(hidden)]
pub use bf16_ranked_module_receipt_v1::{
    ProductionPrivateBf16AttachedRankedOwnerV1, ProductionPrivateBf16FormalMemoryOwnerV1,
};

/// Construction failure before ranked checking starts.
#[derive(Debug)]
pub enum ProductionPreRankedKirErrorV1 {
    /// Existing source, SSA, materialization or correspondence rejection.
    Lowering(ProductionSemanticKirErrorV1),
    /// Connected V12 wire, semantic verification or resource rejection.
    Canonical(fe2o3_kernel_ir::CanonicalKernelIrReplayAdmissionErrorV12),
    /// Exact checked local-frame classification or resource rejection.
    LocalFrame(fe2o3_kernel_ir::LocalFrameErrorV1),
    /// Exact source/SSA occurrence capture or its resource rejection.
    Occurrences(fe2o3_pliron::ProductionSemanticSsaOccurrenceErrorV1),
}

impl fmt::Display for ProductionPreRankedKirErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lowering(error) => error.fmt(formatter),
            Self::Canonical(error) => error.fmt(formatter),
            Self::LocalFrame(error) => error.fmt(formatter),
            Self::Occurrences(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProductionPreRankedKirErrorV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lowering(error) => Some(error),
            Self::Canonical(error) => Some(error),
            Self::LocalFrame(error) => Some(error),
            Self::Occurrences(error) => Some(error),
        }
    }
}

impl From<ProductionSemanticKirErrorV1> for ProductionPreRankedKirErrorV1 {
    fn from(error: ProductionSemanticKirErrorV1) -> Self {
        Self::Lowering(error)
    }
}

impl From<fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1>
    for ProductionPreRankedKirErrorV1
{
    fn from(error: fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1) -> Self {
        Self::Lowering(error.into())
    }
}

/// Helper relation category for routing to supported consumers, not authority
/// to approve a different graph or bypass a source, formal or target check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionHelperSourcePolicyV1 {
    /// Every helper has the existing complete raw-empty effect contract.
    RawEmpty,
    /// At least one helper needs the sealed source-local Unit relation.
    UnitLocal,
    /// Closed nominal BF16 helper; legacy ranked/normal/CPU consumers refuse.
    Bf16Nominal,
}

/// One executable mixed-SSA graph constructed before ranked verification.
///
/// The source plans and launch roster are retained for correspondence and
/// existing source-ranked checks. They are not a second executable graph.
/// This owner does not assert semantic equivalence or grant artifact/launch
/// authority. Explicit later replay audits retain their existing meaning.
///
/// ```compile_fail
/// use fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<ProductionPreRankedKirOwnerV1>();
/// ```
///
/// ```compile_fail
/// use fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1;
/// fn mutate(owner: &mut ProductionPreRankedKirOwnerV1) {
///     owner.executable().module().functions.clear();
/// }
/// ```
#[must_use = "dropping the owner abandons pre-ranked executable custody"]
#[derive(Debug)]
pub struct ProductionPreRankedKirOwnerV1 {
    semantic_ssa: ProductionSemanticSsaOwnerV1,
    executable: fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV12,
    executable_storage: fe2o3_kernel_ir::CanonicalKernelIrReplayStorageV12,
    assert_origins: SealedAssertOriginsV1,
    source_launch: crate::ProductionSourceLaunchRosterV1,
    canonical_kernel_ir: ProductionCanonicalKernelIrV1,
    correspondence: SemanticKirCorrespondenceV1,
    limits: ProductionSemanticKirLimitsV1,
    launch_roots: Box<[RetainedRankedLaunchRootV1]>,
    helper_memory: SealedHelperMemoryV1,
}

/// Borrowed helper facts from one immutable materialization and its validated
/// source/KIR correspondence. Every yielded helper's combined interprocedural
/// effects are complete and pure, including compiler ordering. Helpers admitted
/// under the separate source-local rule are never yielded by this view.
/// This does not establish determinism, termination, or value equivalence.
///
/// ```compile_fail
/// use fe2o3_lower_mir_kernel::{ProductionEmptyEffectHelpersV1, ProductionPreRankedKirOwnerV1};
/// fn forge(owner: &ProductionPreRankedKirOwnerV1) {
///     let _ = ProductionEmptyEffectHelpersV1 { owner };
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ProductionEmptyEffectHelpersV1<'a> {
    owner: &'a ProductionPreRankedKirOwnerV1,
}

impl<'a> ProductionEmptyEffectHelpersV1<'a> {
    /// Work needed to scan the entire function roster, including root rows and
    /// repeated shared-helper associations. The view allocates no payload.
    pub fn scanned_function_count(self) -> usize {
        self.owner.correspondence.lowered_functions().len()
    }

    /// Preserves root owner, semantic function and physical function identity.
    /// Shared helpers can occur more than once, once for each owning root.
    pub fn iter(self) -> impl Iterator<Item = &'a SemanticKirFunctionCorrespondenceV1> {
        self.owner
            .correspondence
            .lowered_functions()
            .iter()
            .enumerate()
            .filter(|(index, row)| {
                row.role() == SemanticKirFunctionRoleV1::InternalHelper
                    && self.owner.helper_memory.is_raw_empty(*index)
            })
            .map(|(_, row)| row)
    }
}

impl ProductionPreRankedKirOwnerV1 {
    /// Materializes the retained SSA plans once, using the full source launch
    /// roster. No ranked verification receipt is accepted at this stage.
    ///
    /// The ledger covers connected V12 admission and assertion-origin emission,
    /// sealing scratch, and retained payload, including their coexistence.
    /// Retained source MIR, SSA plans, launch rows, legacy correspondence/bytes
    /// and other lowering scratch keep their existing limits and are excluded.
    /// Entry-argument replay uses the independent limits in `limits`; its
    /// transient logical payload is not transferred to this canonical ledger.
    /// The incoming live floor is restored after failure drops or success transfer.
    /// Before another allocation, reserve `retained_analysis_storage_v1()` while
    /// this owner or its attached successor lives. Its separate graph, origin
    /// and helper receipts describe those same payloads, not additional copies.
    /// Occurrence capture created for a local helper is included once in that
    /// subtotal; an incoming capture stays separately caller-reserved.
    /// This phase-local logical ledger is not an allocator/RSS meter or a claim
    /// that excluded host allocation failures are recoverable.
    /// This legacy constructor restores the floor on ordinary `Result` return;
    /// it does not provide an unwind-cleanup guarantee. An optional existing SSA
    /// occurrence-capture receipt stays separately caller-reserved and is not
    /// included in the transferred receipts. Pending source-local helpers and
    /// actual workgroup-pipeline creation sites trigger new capture. Its existing
    /// planner/source-replay cost exclusions
    /// remain unchanged; the canonical ledger covers its capture-only payload.
    pub fn try_materialize_with_budget(
        semantic_ssa: ProductionSemanticSsaOwnerV1,
        source_launch: crate::ProductionSourceLaunchRosterV1,
        limits: ProductionSemanticKirLimitsV1,
        budget: &mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'_>,
    ) -> Result<Self, ProductionPreRankedKirErrorV1> {
        let floor = budget.storage();
        let result =
            Self::try_materialize_origins_inner_v1(semantic_ssa, source_launch, limits, budget);
        // The inner call dropped all failed payloads. On success this is the
        // explicit transfer of graph, origin and helper ownership and receipts.
        let release =
            budget
                .storage()
                .checked_sub(floor)
                .ok_or(SemanticKirAssertOriginErrorV1::Resource(
                    AssertOriginResourceV1::Accounting,
                ))?;
        budget
            .release_storage(release)
            .map_err(SemanticKirAssertOriginErrorV1::from)?;
        result
    }

    fn try_materialize_origins_inner_v1(
        semantic_ssa: ProductionSemanticSsaOwnerV1,
        source_launch: crate::ProductionSourceLaunchRosterV1,
        limits: ProductionSemanticKirLimitsV1,
        budget: &mut AssertOriginBudgetV1<'_>,
    ) -> Result<Self, ProductionPreRankedKirErrorV1> {
        Self::try_materialize_origins_with_scalar_capture_v1(
            semantic_ssa,
            source_launch,
            limits,
            None,
            budget,
        )
        .map(|(owner, _)| owner)
    }

    fn try_materialize_origins_with_scalar_capture_v1(
        semantic_ssa: ProductionSemanticSsaOwnerV1,
        source_launch: crate::ProductionSourceLaunchRosterV1,
        limits: ProductionSemanticKirLimitsV1,
        scalar_capture: Option<scalar_ssa_emission_v1::Recorder>,
        budget: &mut AssertOriginBudgetV1<'_>,
    ) -> Result<(Self, Option<scalar_ssa_emission_v1::Recorder>), ProductionPreRankedKirErrorV1>
    {
        Self::try_materialize_origins_with_captures_v1(
            semantic_ssa,
            source_launch,
            limits,
            scalar_capture,
            None,
            budget,
        )
        .map(|(owner, scalar, _)| (owner, scalar))
    }

    fn try_materialize_origins_with_captures_v1(
        mut semantic_ssa: ProductionSemanticSsaOwnerV1,
        source_launch: crate::ProductionSourceLaunchRosterV1,
        limits: ProductionSemanticKirLimitsV1,
        scalar_capture: Option<scalar_ssa_emission_v1::Recorder>,
        tiled_capture: Option<tiled_region_v1::Capture>,
        budget: &mut AssertOriginBudgetV1<'_>,
    ) -> Result<
        (
            Self,
            Option<scalar_ssa_emission_v1::Recorder>,
            Option<tiled_region_v1::Capture>,
        ),
        ProductionPreRankedKirErrorV1,
    > {
        let launch_roots = materialization_launch_roots_v1(&semantic_ssa, &source_launch)?;
        budget.charge_work(2)?;
        let mut capture = match semantic_ssa.occurrence_storage() {
            Some(receipt) => {
                if budget.storage() < receipt.retained_storage() {
                    return Err(HelperMemoryResourceV1::Accounting.into());
                }
                HelperOccurrenceCaptureV1::Preexisting(receipt)
            }
            None => HelperOccurrenceCaptureV1::Absent,
        };
        if matches!(capture, HelperOccurrenceCaptureV1::Absent) && ordinary_source_has_raw_wave64_shuffle_v1(semantic_ssa.source_semantic()) {
            let receipt=semantic_ssa.try_capture_occurrences_with_budget_v1(budget).map_err(ProductionPreRankedKirErrorV1::Occurrences)?;
            budget.reserve_storage(receipt.retained_storage())?;
            capture=HelperOccurrenceCaptureV1::Transferred(receipt);
        }
        let mut emitted_origins = AssertOriginEmissionV1::new(budget);
        emitted_origins.scalar_capture = scalar_capture;
        emitted_origins.tiled_capture = tiled_capture;
        let PendingHelperSourceLoweringV1 {
            module,
            correspondence,
            requires_source,
        } = lower_pending_module_with_assert_origins_v1(
            &semantic_ssa,
            limits,
            &launch_roots,
            &mut emitted_origins,
        )?;
        // Keep the frozen legacy bytes for existing publication/replay consumers.
        // Their transient inverse validation is not another retained graph.
        let canonical_kernel_ir = ProductionCanonicalKernelIrV1::from_module_ref(&module)?;
        let (executable, executable_storage) =
            fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV12::
                from_module_ref_with_verification_budget_v12(&module, emitted_origins.budget)
                .map_err(ProductionPreRankedKirErrorV1::Canonical)?;
        emitted_origins
            .budget
            .reserve_storage(executable_storage.retained_storage())
            .map_err(SemanticKirAssertOriginErrorV1::from)?;
        drop(module);
        if matches!(capture, HelperOccurrenceCaptureV1::Absent)
            && (requires_source
                || pre_ranked_pipeline_occurrences_required_v1(
                    &semantic_ssa,
                    emitted_origins.budget,
                )?)
        {
            let receipt = semantic_ssa
                .try_capture_occurrences_with_budget_v1(emitted_origins.budget)
                .map_err(ProductionPreRankedKirErrorV1::Occurrences)?;
            emitted_origins
                .budget
                .reserve_storage(receipt.retained_storage())?;
            capture = HelperOccurrenceCaptureV1::Transferred(receipt);
        }
        let scalar_capture = emitted_origins.scalar_capture.take();
        let tiled_capture = emitted_origins.tiled_capture.take();
        let assert_origins = emitted_origins.seal(&semantic_ssa, &correspondence, &executable)?;
        let subject = CanonicalCallSubjectV1 {
            semantic_ssa: &semantic_ssa,
            executable: &executable,
            correspondence: &correspondence,
        };
        let mut helper_memory = if requires_source {
            SealedHelperMemoryV1::derive_with_origins_v1(subject, Some(&assert_origins), budget)?
        } else {
            SealedHelperMemoryV1::derive(subject, budget)?
        };
        budget
            .charge_work(4)
            .map_err(ProductionSemanticKirErrorV1::from)?;
        let has_checked_source = !helper_memory.unit_source.is_empty();
        if requires_source != has_checked_source {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch.into());
        }
        helper_memory.capture = capture;
        helper_memory.analysis_storage = helper_memory_live_storage_v1(
            executable_storage.retained_storage(),
            assert_origins.storage.payload_storage(),
            helper_memory.storage.retained_storage(),
        )?
        .checked_add(capture.transferred_storage())
        .ok_or(HelperMemoryResourceV1::Arithmetic)?;
        Ok((
            Self {
                semantic_ssa,
                executable,
                executable_storage,
                assert_origins,
                source_launch,
                canonical_kernel_ir,
                correspondence,
                limits,
                launch_roots,
                helper_memory,
            },
            scalar_capture,
            tiled_capture,
        ))
    }

    /// Borrows the exact source and SSA plans consumed by materialization.
    pub const fn semantic_ssa(&self) -> &ProductionSemanticSsaOwnerV1 {
        &self.semantic_ssa
    }

    /// Borrows the only retained executable graph and its canonical V12 bytes.
    pub const fn executable(&self) -> &fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV12 {
        &self.executable
    }

    /// Borrows only facts established during this owner's exact materialization.
    pub const fn empty_effect_helpers(&self) -> ProductionEmptyEffectHelpersV1<'_> {
        ProductionEmptyEffectHelpersV1 { owner: self }
    }

    /// Returns only a consumer-routing category. The typed source relation must
    /// still be borrowed and checked by any stage that supports local helpers.
    pub fn helper_source_policy_v1(&self) -> ProductionHelperSourcePolicyV1 {
        if self.helper_memory.bf16_nominal.is_some() {
            ProductionHelperSourcePolicyV1::Bf16Nominal
        } else if self.helper_memory.unit_source.is_empty() {
            ProductionHelperSourcePolicyV1::RawEmpty
        } else {
            ProductionHelperSourcePolicyV1::UnitLocal
        }
    }

    fn require_legacy_helper_policy_v1(
        &self,
        consumer: &'static str,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if self.helper_source_policy_v1() != ProductionHelperSourcePolicyV1::RawEmpty {
            return Err(
                ProductionSemanticKirErrorV1::LocalHelperSourceConsumerUnavailable { consumer },
            );
        }
        Ok(())
    }

    /// Returns the connected graph payload to reserve when continuing its ledger.
    pub const fn executable_storage(&self) -> fe2o3_kernel_ir::CanonicalKernelIrReplayStorageV12 {
        self.executable_storage
    }

    /// Retained helper payload, to reserve with the graph and assertion origins.
    pub const fn helper_memory_storage_v1(&self) -> ProductionHelperMemoryStorageV1 {
        self.helper_memory.storage
    }

    /// Complete retained graph, assertion-origin, helper and newly created SSA
    /// occurrence payload. Incoming capture keeps its separate caller reservation.
    pub const fn retained_analysis_storage_v1(&self) -> usize {
        self.helper_memory.analysis_storage
    }

    /// Borrows source assertion origins sealed against this exact executable.
    pub const fn assert_origins(&self) -> SemanticKirAssertOriginsV1<'_> {
        SemanticKirAssertOriginsV1 {
            executable: &self.executable,
            semantic_ssa: &self.semantic_ssa,
            origins: &self.assert_origins,
        }
    }

    /// Returns the separate retained assertion-origin payload transfer.
    pub const fn assert_origin_storage(&self) -> SemanticKirAssertOriginStorageV1 {
        self.assert_origins.storage
    }

    /// Borrows the complete source-only launch roster used during materialization.
    pub const fn source_launch(&self) -> &crate::ProductionSourceLaunchRosterV1 {
        &self.source_launch
    }

    /// Compiler custody grants no artifact, verification-policy or launch authority.
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

// The ranked pipeline projector consumes original use-site SSA, not merely
// lowered scalar types. A declared-but-unused intrinsic creates no demand.
// This scan borrows the already admitted source and allocates no side table.
fn pre_ranked_pipeline_occurrences_required_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    budget: &mut AssertOriginBudgetV1<'_>,
) -> Result<bool, ProductionPreRankedKirErrorV1> {
    let source = owner.source_semantic();
    for function in source.functions() {
        budget.charge_work(1)?;
        for block in function.blocks() {
            budget.charge_work(1)?;
            let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
                continue;
            };
            budget.charge_work(1)?;
            let callable = source
                .callables()
                .get(call.callee().index() as usize)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            if matches!(
                callable,
                SemanticCallableDeclV1::CompilerIntrinsic {
                    operation: SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineCreate { .. },
                    ..
                }
            ) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Ranked checks attached to the exact previously materialized graph.
/// Unlike the legacy receipt this cannot cause source re-materialization.
#[must_use = "dropping the receipt abandons ranked and executable custody"]
pub struct ProductionMaterializedRankedModuleReceiptV1 {
    materialized: ProductionPreRankedKirOwnerV1,
    roots: Box<[ProductionRankedSemanticProjectionRootV1]>,
}

impl ProductionMaterializedRankedModuleReceiptV1 {
    /// Checks the complete root bijection and exact source execution layout,
    /// then validates existing borrowed ranked correspondence against this owner.
    pub fn from_unvalidated_projection_roster_candidate(
        materialized: ProductionPreRankedKirOwnerV1,
        roots: Vec<ProductionRankedSemanticProjectionRootV1>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        materialized.require_legacy_helper_policy_v1("materialized ranked receipt")?;
        validate_source_ranked_roster_v1(
            &materialized.semantic_ssa,
            &materialized.source_launch,
            &roots,
        )?;
        Ok(Self {
            materialized,
            roots: roots.into_boxed_slice(),
        })
    }

    /// Returns the retained complete ranked root count.
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }
    /// Compiler custody grants no artifact, verification-policy or launch authority.
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

#[derive(Debug)]
enum RetainedProductionKirModuleV1 {
    Legacy(Module),
    Connected {
        executable: fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV12,
        storage: fe2o3_kernel_ir::CanonicalKernelIrReplayStorageV12,
        assert_origins: SealedAssertOriginsV1,
        source_launch: crate::ProductionSourceLaunchRosterV1,
        helper_memory: SealedHelperMemoryV1,
    },
}

impl RetainedProductionKirModuleV1 {
    const fn module(&self) -> &Module {
        match self {
            Self::Legacy(module) => module,
            Self::Connected { executable, .. } => executable.module(),
        }
    }
}

impl std::ops::Deref for RetainedProductionKirModuleV1 {
    type Target = Module;
    fn deref(&self) -> &Module {
        self.module()
    }
}

impl ProductionSemanticKirOwnerV1 {
    /// Attaches checked ranked custody without invoking executable lowering.
    /// Existing `verify_equivalence` remains an explicit reconstruction audit.
    /// Graph, origin and helper receipts remain caller-reserved across this consuming
    /// attachment. Any preexisting SSA occurrence-capture receipt is separate
    /// and remains reserved until its source owner is dropped, including when
    /// receipt validation or attachment fails.
    pub fn try_attach_materialized_ranked_checks(
        receipt: ProductionMaterializedRankedModuleReceiptV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let limits = receipt.materialized.limits;
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(
            limits.max_argument_correspondence_work,
        );
        let mut budget =
            ArgumentBudgetV1::new(&mut work, limits.max_argument_correspondence_storage);
        Self::try_attach_materialized_ranked_checks_with_budget_v1(receipt, &mut budget)
    }

    /// Attaches the same exact materialized source/ranked custody with new
    /// helper-value cache and expansion work charged to the caller's ledger.
    /// Existing graph, origin, capture and source floors remain caller-owned.
    pub fn try_attach_materialized_ranked_checks_with_budget_v1(
        receipt: ProductionMaterializedRankedModuleReceiptV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        Self::try_attach_materialized_ranked_checks_using_allowance_v1(receipt, budget, None)
    }

    /// Same consuming attachment and original caller ledger, additionally
    /// preserving the materialized owner's legacy local translation work and
    /// live-scratch ceilings across the complete root roster. Only native helper
    /// scan/cache/expansion is charged; graph/canonical/receipt costs stay excluded.
    pub fn try_attach_materialized_ranked_checks_with_bounded_translation_budget_v1(
        receipt: ProductionMaterializedRankedModuleReceiptV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let limits = receipt.materialized.limits;
        let mut allowance = native_helper_value_expansion_v1::TranslationAllowanceV1::new(
            budget,
            limits.max_argument_correspondence_work,
            limits.max_argument_correspondence_storage,
        );
        Self::try_attach_materialized_ranked_checks_using_allowance_v1(
            receipt,
            budget,
            Some(&mut allowance),
        )
    }

    fn try_attach_materialized_ranked_checks_using_allowance_v1(
        receipt: ProductionMaterializedRankedModuleReceiptV1,
        budget: &mut ArgumentBudgetV1<'_>,
        mut allowance: Option<&mut native_helper_value_expansion_v1::TranslationAllowanceV1>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        receipt
            .materialized
            .require_legacy_helper_policy_v1("ranked attachment")?;
        let ProductionMaterializedRankedModuleReceiptV1 {
            materialized,
            roots,
        } = receipt;
        let ProductionPreRankedKirOwnerV1 {
            semantic_ssa,
            executable,
            executable_storage,
            assert_origins,
            source_launch,
            canonical_kernel_ir,
            correspondence,
            limits,
            launch_roots,
            helper_memory,
        } = materialized;
        let semantic = semantic_ssa.source_semantic();
        let mut generic_checks = Vec::with_capacity(roots.len());
        for root in roots.into_vec() {
            let function_name = root.function_name().to_owned();
            let translation_validation =
                validate_mir_pliron_translation_with_allowance_resources_v1(
                    Some(&semantic_ssa),
                    executable.module(),
                    &correspondence,
                    &function_name,
                    &root.lowering,
                    &root.access_sources,
                    &root.executable_effect_sources,
                    limits.max_operations,
                    budget,
                    allowance.as_deref_mut(),
                )
                .map_err(
                    native_helper_value_expansion_v1::NativeTranslationErrorV1::into_semantic,
                )?;
            generic_checks.push(RetainedGenericKernelChecksV1 {
                selected_root: root.selected_root,
                launch_rank: root.launch_rank,
                semantic_sha256: *semantic.semantic_sha256().as_bytes(),
                function_name,
                ranked_ir: root.ranked_ir.into_boxed_str(),
                lowering: root.lowering,
                access_sources: root.access_sources,
                executable_effect_sources: root.executable_effect_sources,
                translation_validation,
            });
        }
        Ok(Self {
            semantic_ssa,
            module: RetainedProductionKirModuleV1::Connected {
                executable,
                storage: executable_storage,
                assert_origins,
                source_launch,
                helper_memory,
            },
            canonical_kernel_ir,
            correspondence,
            limits,
            launch_roots: Some(launch_roots),
            generic_checks: generic_checks.into_boxed_slice(),
        })
    }

    /// The exact pre-ranked graph, when construction used the new production
    /// stage. Legacy standalone constructors do not synthesize this custody.
    pub const fn pre_ranked_executable(
        &self,
    ) -> Option<&fe2o3_kernel_ir::VerifiedCanonicalKernelIrModuleV12> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected { executable, .. } => Some(executable),
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }

    /// Returns the retained connected graph's original allocation receipt.
    pub const fn pre_ranked_executable_storage(
        &self,
    ) -> Option<fe2o3_kernel_ir::CanonicalKernelIrReplayStorageV12> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected { storage, .. } => Some(*storage),
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }

    /// Borrows retained assertion origins only on the connected production path.
    pub const fn pre_ranked_assert_origins(&self) -> Option<SemanticKirAssertOriginsV1<'_>> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected {
                executable,
                assert_origins,
                ..
            } => Some(SemanticKirAssertOriginsV1 {
                executable,
                semantic_ssa: &self.semantic_ssa,
                origins: assert_origins,
            }),
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }

    /// Returns the separate retained-origin floor alongside the graph receipt.
    pub const fn pre_ranked_assert_origin_storage(
        &self,
    ) -> Option<SemanticKirAssertOriginStorageV1> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected { assert_origins, .. } => {
                Some(assert_origins.storage)
            }
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }

    /// Retained helper obligations move with connected graph custody. Legacy
    /// owners do not synthesize this receipt.
    pub const fn pre_ranked_helper_memory_storage_v1(
        &self,
    ) -> Option<ProductionHelperMemoryStorageV1> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected { helper_memory, .. } => {
                Some(helper_memory.storage)
            }
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }

    /// Complete checked retained subtotal transferred with a connected owner.
    /// Source SSA occurrence capture remains a separate caller reservation.
    pub const fn pre_ranked_retained_analysis_storage_v1(&self) -> Option<usize> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected { helper_memory, .. } => {
                Some(helper_memory.analysis_storage)
            }
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }

    /// Borrows the source-only launch roster retained by the new production path.
    pub const fn source_launch_roster(&self) -> Option<&crate::ProductionSourceLaunchRosterV1> {
        match &self.module {
            RetainedProductionKirModuleV1::Connected { source_launch, .. } => Some(source_launch),
            RetainedProductionKirModuleV1::Legacy(_) => None,
        }
    }
}

impl ProductionCanonicalKernelIrV1 {
    fn from_module_ref(module: &Module) -> Result<Self, ProductionSemanticKirErrorV1> {
        macro_rules! canonicalize {
            ($encode:path, $owner:ty, $variant:ident, $error:ident, $canonical_error:ident) => {{
                let bytes = $encode(module).map_err(|error| {
                    ProductionSemanticKirErrorV1::$error($canonical_error::Encode(error))
                })?;
                let (owner, inverse) = <$owner>::from_canonical_bytes_with_module(bytes)
                    .map_err(ProductionSemanticKirErrorV1::$error)?;
                if &inverse != module {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                Ok(Self::$variant(owner))
            }};
        }
        if module_requires_kernel_ir_v11_v1(module) {
            canonicalize!(
                fe2o3_kernel_ir::encode_module_v11,
                VerifiedCanonicalKernelIrV11,
                V11,
                CanonicalKernelIrV11,
                VerifiedCanonicalKernelIrErrorV11
            )
        } else if module_requires_kernel_ir_v9_v1(module) {
            canonicalize!(
                fe2o3_kernel_ir::encode_module_v9,
                VerifiedCanonicalKernelIrV9,
                V9,
                CanonicalKernelIrV9,
                VerifiedCanonicalKernelIrErrorV9
            )
        } else {
            canonicalize!(
                fe2o3_kernel_ir::encode_module_v8,
                VerifiedCanonicalKernelIrV8,
                V8,
                CanonicalKernelIrV8,
                VerifiedCanonicalKernelIrErrorV8
            )
        }
    }
}

fn materialization_launch_roots_v1(
    semantic_ssa: &ProductionSemanticSsaOwnerV1,
    source_launch: &crate::ProductionSourceLaunchRosterV1,
) -> Result<Box<[RetainedRankedLaunchRootV1]>, ProductionSemanticKirErrorV1> {
    semantic_ssa
        .verify_replay()
        .map_err(ProductionSemanticKirErrorV1::SemanticSsa)?;
    let semantic = semantic_ssa.source_semantic();
    if source_launch.semantic_sha256() != semantic.semantic_sha256().as_bytes()
        || source_launch.roots().len() != semantic.roots().len()
        || source_launch
            .roots()
            .iter()
            .zip(semantic.roots())
            .any(|(row, root)| row.selected_root() != *root)
    {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    Ok(source_launch
        .roots()
        .iter()
        .map(|root| {
            let layout = root.layout();
            RetainedRankedLaunchRootV1 {
                selected_root: root.selected_root(),
                launch_rank: root.source_rank(),
                global_extents: layout.global_extents(),
                workgroup_extents: layout.workgroup_extents(),
                full_physical_workgroups: layout.full_physical_workgroups(),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn validate_source_ranked_roster_v1(
    semantic_ssa: &ProductionSemanticSsaOwnerV1,
    source_launch: &crate::ProductionSourceLaunchRosterV1,
    roots: &[ProductionRankedSemanticProjectionRootV1],
) -> Result<(), ProductionSemanticKirErrorV1> {
    semantic_ssa
        .verify_replay()
        .map_err(ProductionSemanticKirErrorV1::SemanticSsa)?;
    if roots.is_empty() || roots.len() != source_launch.roots().len() {
        return Err(unsupported(
            0,
            None,
            None,
            "ranked projection roster is not a complete semantic root bijection",
        ));
    }
    let mut names = BTreeSet::new();
    for (root, source) in roots.iter().zip(source_launch.roots()) {
        if root.selected_root != source.selected_root()
            || root.launch_rank != source.source_rank()
            || !names.insert(root.function_name())
        {
            return Err(unsupported(
                root.selected_root.index(),
                None,
                None,
                "ranked projection roster has a duplicate, missing, or invalid root identity",
            ));
        }
        let layout = source.layout();
        let expected = ProductionRankedOperationV1::ExecutionLayout {
            grid_identity: layout.grid_identity(),
            global_extents: layout.global_extents(),
            workgroup_extents: layout.workgroup_extents(),
            subgroup_size: layout.subgroup_size(),
            full_physical_workgroups: layout.full_physical_workgroups(),
        };
        if root
            .lowering
            .kernel()
            .blocks()
            .first()
            .and_then(|block| block.operations().first())
            != Some(&expected)
        {
            return Err(unsupported(
                root.selected_root.index(),
                None,
                None,
                "ranked execution layout changed after executable materialization",
            ));
        }
        validate_borrowed_ranked_semantic_projection_candidate_with_generated_effects_v1(
            semantic_ssa.source_owner(),
            root.selected_root,
            &root.lowering,
            &root.ranked_ir,
            &root.access_sources,
            &root.executable_effect_sources,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "production_pre_ranked_legacy_tests_v1.rs"]
mod pre_ranked_legacy_tests_v1;
