use super::cpu_origin::DecodedCpu;
use super::*;
#[cfg(test)]
use crate::portable_reference_v1::codec::NativeCpuAssociationV1;
use fe2o3_functional_proof::{
    FunctionalRefinementBoundaryV2, FunctionalRefinementImportPolicyV2,
    ImportedFunctionalRefinementProofV2,
};
use fe2o3_lower_mir_kernel::{
    ProductionRankedSourceRowsV1, decode_production_ranked_source_rows_v1,
    with_conditional_root_request_v1,
};
use fe2o3_pliron::{
    ProductionConstructionV1, ProductionPlironSessionV1, ProductionRefinementStagingPolicyV2,
    ProductionSessionLimitsV1, stage_ranked_kernel_with_borrowed_policy_checked_refinement_v2,
};

#[cfg(test)]
pub(super) fn association(
    cpu: NativeCpuAssociationV1<'_>,
    semantic: [u8; 32],
    row: &NativeConditionalSourceRootV2<'_>,
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    association_fields(
        cpu.semantic_mir_sha256,
        cpu.semantic_root,
        cpu.logical_kernel_name,
        semantic,
        row,
        budget,
    )
}

fn association_fields(
    cpu_semantic: [u8; 32],
    cpu_root: u32,
    cpu_name: &str,
    semantic: [u8; 32],
    row: &NativeConditionalSourceRootV2<'_>,
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    budget.charge_work(
        cpu_name
            .len()
            .checked_add(row.launch.logical_name().len())
            .and_then(|n| n.checked_add(34))
            .ok_or(Resource::Arithmetic)?,
    )?;
    if cpu_semantic != semantic
        || cpu_root != row.semantic_root
        || cpu_name != row.launch.logical_name()
    {
        return Err(E::invalid(
            "conditional CPU source/root/logical-name association",
        ));
    }
    // Registration origin and launch max_grid cannot be established from MIR.
    Ok(())
}

pub(super) fn accepted_effect(
    signature: &crate::InertFunctionalRefinementReceiptSignatureV2,
    expected: &crate::NativeCompilerStagingCommitmentV1,
    policy: &ProductionRefinementStagingPolicyV2,
    mut charge: impl FnMut(usize) -> Result<(), Resource>,
) -> Result<(), crate::NativeCompilerSourceProofErrorV1> {
    use crate::NativeCompilerSourceProofErrorV1 as OldError;
    charge(65)?;
    let identified = FunctionalRefinementImportPolicyV2::new(
        *signature.verifying_key(),
        policy.toolchain(),
        FunctionalRefinementBoundaryV2::SafeReferenceMirToKernelMir,
    )
    .map_err(OldError::EffectReceipt)?;
    // Identifying the transported key does not accept it. Membership is in the
    // external policy, before the shared ordinal resolver imports any signature.
    if !policy.accepts_signer(identified.signer_identity())
        || expected.signer != *identified.signer_identity().as_bytes()
    {
        return Err(OldError::Mismatch(
            "externally accepted conditional effect signer",
        ));
    }
    Ok(())
}

pub(super) fn reconstruct_root(
    source: &ReplayedNativeSourceV1,
    row: &NativeConditionalSourceRootV2<'_>,
    policy: &NativeConditionalRootPolicyV2<'_>,
    expected: cpu_mapping::RootExpectation<'_>,
    budget: &mut Budget<'_>,
) -> Result<ReplayedRoot, E> {
    reconstruct_root_using(
        source,
        row,
        policy,
        expected,
        budget,
        |request, decoded, budget| match decoded {
            DecodedCpu::Registration(decoded) => {
                crate::import_and_retain_conditional_ranked_formula_v2(
                    request,
                    decoded,
                    row.formula_receipt,
                    policy.formula,
                    budget,
                )
            }
            DecodedCpu::Policy(decoded) => {
                crate::import_and_retain_conditional_ranked_formula_policy_v2(
                    request,
                    decoded,
                    row.formula_receipt,
                    policy.formula,
                    budget,
                )
            }
        },
        |formula| formula.map_err(|error| E(Cause::Formula(error))),
    )
}

pub(super) fn reconstruct_root_using<R, Failure: From<E> + From<Resource>>(
    source: &ReplayedNativeSourceV1,
    row: &NativeConditionalSourceRootV2<'_>,
    policy: &NativeConditionalRootPolicyV2<'_>,
    expected: cpu_mapping::RootExpectation<'_>,
    budget: &mut Budget<'_>,
    import: impl FnOnce(
        &fe2o3_lower_mir_kernel::ProductionSourceBoundConditionalAggregateRequestV1<'_>,
        DecodedCpu<'_>,
        &mut Budget<'_>,
    ) -> R,
    finish: impl FnOnce(R) -> Result<RetainedProductionConditionalFormulaV2, Failure>,
) -> Result<ReplayedRoot, Failure> {
    cpu_origin::with_decoded(
        row.cpu_input_bytes,
        expected.origin,
        budget,
        |decoded, budget| {
            let cpu = decoded.subjects();
            association_fields(
                cpu.semantic_mir_sha256,
                cpu.semantic_root,
                cpu.logical_kernel_name,
                *source
                    .source()
                    .semantic_ssa()
                    .source_semantic()
                    .semantic_sha256()
                    .as_bytes(),
                row,
                budget,
            )?;
            expected.require_subjects(&cpu, budget)?;
            let subjects =
                crate::conditional_reference_v1::reference_subjects_v1(cpu.kernel, cpu.reference)
                    .map_err(|error| E(Cause::Correspondence(error)))?;
            let pending = pending(row, policy.effects, budget)?;
            let (rows, storage) =
                decode_production_ranked_source_rows_v1(row.source_rows_bytes, budget)
                    .map_err(|error| E(Cause::Rows(error)))?;
            budget.reserve_storage(storage.retained_storage())?;
            let (access_sources, executable_effect_sources) = rows.into_parts();
            let ranked_ir = account::text(row.ranked_ir, budget)?;
            let input = ProductionConditionalRootInputV1 {
                pending,
                semantic_root: row.semantic_root,
                launch_rank: row.launch_rank,
                access_sources,
                executable_effect_sources,
                ranked_ir,
                reference_subjects: subjects,
            };
            // Inline headers live in the root vector's already-prepaid slot.
            budget.release_storage(size_of::<ProductionRankedSourceRowsV1>())?;
            let prior = input.retained_storage_v1()?;
            let floor = budget.storage();
            let ledger = budget.work_ledger_identity_v1();
            let result = with_conditional_root_request_v1(
                source.source(),
                input,
                budget,
                |request, budget| import(request, decoded, budget),
            );
            // The lower continuation reserves its returned input anew. Only after
            // every lower postcheck succeeds can its original reservation transfer.
            if budget.work_ledger_identity_v1() != ledger || budget.storage() < floor {
                drop(result);
                return Err(Resource::Accounting.into());
            }
            let (formula, input) = result.map_err(|error| E(Cause::Continuation(error)))?;
            let formula = finish(formula)?;
            budget.release_storage(prior)?;
            Ok(ReplayedRoot { input, formula })
        },
    )
}

fn pending(
    row: &NativeConditionalSourceRootV2<'_>,
    accepted: &ProductionRefinementStagingPolicyV2,
    budget: &mut Budget<'_>,
) -> Result<fe2o3_pliron::ProductionConditionalRankedAnalysisV1, E> {
    let headers = size_of::<Vec<ImportedFunctionalRefinementProofV2>>();
    budget.reserve_storage(headers)?;
    let (mut proofs, proof_storage) = account::vector(row.effect_receipts.len(), budget)?;
    let (kernel, recipe_storage) =
        crate::compiler_native_source_proof_v1::decode_signed_recipe_with_import_hooks_v1(
            row.recipe_bytes,
            row.effect_receipts,
            row.staging_commitments,
            accepted.toolchain(),
            budget,
            |signature, expected, work| {
                accepted_effect(signature, expected, accepted, |n| work.charge_work(n))
            },
            |proof| proofs.push(proof),
        )
        .map_err(|error| E(Cause::Recipe(error)))?;
    budget.reserve_storage(recipe_storage.retained_storage())?;
    let site = select_native_conditional_ownership_site_v2(&kernel, budget)?;
    // Preserve staging's own duplicate-claim rules. Repeated accepted signers
    // are normal; no extra unique signer or digest roster is invented here.
    let construction = ProductionConstructionV1::ranked_kernel("conditional_source_replay", kernel)
        .map_err(|_| E::invalid("conditional registered construction name"))?;
    let construction = stage_ranked_kernel_with_borrowed_policy_checked_refinement_v2(
        construction,
        proofs,
        accepted,
    )
    .map_err(|error| E(Cause::Staging(error)))?;
    let mut session =
        ProductionPlironSessionV1::new_ranked_v1(ProductionSessionLimitsV1::default())
            .map_err(|error| E(Cause::SessionCreate(error)))?;
    let construction = session
        .register_construction(construction)
        .map_err(|error| E(Cause::Session(error)))?;
    let (stage, root) = session
        .construct_registered(construction)
        .map_err(|error| E(Cause::Session(error)))?;
    let pending = session
        .prepare_conditional_ranked_analysis_v1(stage, root, &[site])
        .map_err(|error| E(Cause::Session(error)))?;
    // These inherited Pliron operations use the same bounded session domain as
    // production. Keep decoded recipe/proof charges until actual arena transfer.
    budget.reserve_storage(pending.retained_analysis_storage_v1())?;
    budget.release_storage(
        headers
            .checked_add(proof_storage)
            .and_then(|n| n.checked_add(recipe_storage.retained_storage()))
            .ok_or(Resource::Arithmetic)?,
    )?;
    Ok(pending)
}
