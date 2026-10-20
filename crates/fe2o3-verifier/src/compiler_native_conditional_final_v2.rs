//! Conditional source-through-F content only; no final/nominal/native authority.
#![allow(
    clippy::result_large_err,
    clippy::large_enum_variant,
    reason = "typed terminal errors without an uncharged allocation"
)]
use super::NativeConditionalCpuExpectationV1;
use super::cpu_mapping::{
    NativeConditionalCpuMappingContextV1, ProjectedSelection, ReplaySelection,
};
use super::cpu_origin::{CpuReplayMode, DecodedCpu};
use super::{
    Budget, E as SourceError, NativeConditionalRootPolicyV2, NativeConditionalSourcePacketInputV2,
    NativeConditionalSourceStorageV2, ReplayedNativeConditionalSourceV2, ReplayedNativeSourceV1,
    ReplayedRoot, Resource, reconstruct, root, with_decoded_native_conditional_source_packet_v2,
};
use crate::conditional_contract_request_v2::check_conditional_contract_content_v2;
use crate::conditional_ranked_formulas_v1::{
    import_and_check_conditional_ranked_formula_policy_v2,
    import_and_check_conditional_ranked_formula_v2,
};
use fe2o3_amd_target::ProductionAmdTargetProfileV1 as Profile;
use fe2o3_amdgcn_model::{
    ReplayedNativeV12TextDescriptorRelationV5 as Relation,
    check_native_v12_text_descriptor_relation_v5,
    check_production_target_coordinate_preservation_v1,
};
use fe2o3_kernel_descriptor::{
    DESCRIPTOR_TABLE_VIEW_STORAGE_V5, DeviceDescriptorTableV5, KernelId,
};
use fe2o3_kernel_ir::InertCanonicalKernelIrContractCatalogV1 as Catalog;
use fe2o3_kernel_opt::{
    CanonicalRefinedForwardingHistoryLimitsV1 as Limits,
    CheckedCanonicalRefinedForwardingHistoryV1 as History,
    DecodedRefinedForwardingHistoryV1 as DecodedHistory, RefinedForwardingHistoryRoleV1 as Role,
};
use fe2o3_mir_model::semantic_mir_v1::SemanticFunctionIdV1;
use std::mem::size_of;

#[path = "compiler_native_conditional_final_v2/account.rs"]
pub(crate) mod account;
#[path = "compiler_native_conditional_final_v2/error.rs"]
mod error;
#[path = "compiler_native_conditional_final_v2/manifest.rs"]
pub(crate) mod manifest;
#[path = "compiler_native_conditional_final_v2/roster.rs"]
mod roster;
pub use error::NativeConditionalFinalErrorV2;
use error::{Cause, NativeConditionalFinalErrorV2 as Error};

/// All backing is caller-owned and prepaid on the original consumer ledger.
/// Limits and profile are independently accepted, never copied from transport.
/// Neither these inputs nor their content agreement authenticates source origin,
/// original nominal/context custody, registration or launch restrictions.
pub struct NativeConditionalFinalInputsV2<'a, 'frame, 'wire> {
    /// Independently materialized complete history from the actual retained chain.
    pub decoded_history: &'a DecodedHistory<'frame, 'wire>,
    /// Original caller limits, not values selected from the transported history.
    pub expected_limits: Limits,
    /// Exact paid contract catalog for final F.
    pub final_catalog: &'a Catalog,
    /// Exact canonical final graph bytes, retained separately or borrowed from F.
    pub published_output_bytes: &'a [u8],
    /// Independently accepted target profile.
    pub profile: Profile,
    /// Paid V5 table borrowing its complete canonical source bytes.
    pub descriptors: &'a DeviceDescriptorTableV5<'wire>,
    /// Complete canonical LLVM with its exact V5 descriptor suffix.
    pub final_llvm: &'a str,
}
type Inputs<'a, 'frame, 'wire> = NativeConditionalFinalInputsV2<'a, 'frame, 'wire>;
type Output = (
    ReplayedNativeConditionalSourceV2,
    NativeConditionalSourceStorageV2,
);
// Closed-route peak: transfer + HEADER + history + coordinates + either roster
// query or final relation. The two leaf scopes never overlap; temporary() pays
// only its requested payload, so all five Account headers must be prepaid here.
const HEADER: usize = size_of::<Inputs<'static, 'static, 'static>>()
    + 5 * size_of::<account::Account>()
    + size_of::<Result<Output, Error>>();

/// One checked history, one N/B relation and one B1/lower/V2 import per root.
/// Returns only existing C1 source content custody with UNRESERVED storage.
/// The caller still owns decoded actual F; no F/nominal/launch owner is minted.
/// All errors (including overriding postchecks) are terminal for refund purposes.
/// Producer custody, original nominal/context identity, registration and launch
/// restrictions remain mandatory enclosing obligations, not content authority.
pub fn validate_native_conditional_source_through_f_v2(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    validate_native_conditional_source_through_f_using_v2(
        packet_bytes,
        accepted,
        inputs,
        budget,
        |_, _, _, _| Ok(()),
    )
}

/// Explicit per-root codec selection from independently retained expectations.
/// Expectations are inert, caller-owned and prepaid on the original account;
/// their agreement does not authenticate original enrollment or source custody.
pub fn validate_native_conditional_source_through_f_with_cpu_origins_v2(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_cpu: &[NativeConditionalCpuExpectationV1],
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    validate_using(
        packet_bytes,
        accepted,
        CpuReplayMode::Expected(expected_cpu).into(),
        inputs,
        budget,
        false,
        |_, _, _, _| Ok(()),
    )
}

fn header<Failure>(callback: usize) -> Result<usize, Resource> {
    HEADER
        .checked_sub(size_of::<Result<Output, Error>>())
        .and_then(|n| n.checked_add(size_of::<Result<Output, Failure>>()))
        .and_then(|n| n.checked_add(callback))
        .ok_or(Resource::Arithmetic)
}

/// The join sees the same checked relation before C0/account postchecks. Its
/// unit result cannot transfer borrowed custody. All failures stay terminal.
pub(crate) fn validate_native_conditional_source_through_f_using_v2<Failure>(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<Output, Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    validate_using(
        packet_bytes,
        accepted,
        CpuReplayMode::RegistrationOnly.into(),
        inputs,
        budget,
        false,
        join,
    )
}

/// Selected only by the concrete original-account V5 recovery window. Nested
/// history composition counts its complete original owners again; no source
/// reconstruction, callback or terminal account rule is replaced here.
pub(crate) fn validate_native_conditional_source_through_f_using_original_account_v2<Failure>(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<Output, Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    validate_using(
        packet_bytes,
        accepted,
        CpuReplayMode::RegistrationOnly.into(),
        inputs,
        budget,
        true,
        join,
    )
}

pub(crate) fn validate_native_conditional_source_through_f_with_cpu_origins_using_v2<Failure>(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_cpu: &[NativeConditionalCpuExpectationV1],
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<Output, Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    validate_using(
        packet_bytes,
        accepted,
        CpuReplayMode::Expected(expected_cpu).into(),
        inputs,
        budget,
        false,
        join,
    )
}

pub(crate) fn validate_native_conditional_source_through_f_with_cpu_origins_using_original_account_v2<
    Failure,
>(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    expected_cpu: &[NativeConditionalCpuExpectationV1],
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<Output, Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    validate_using(
        packet_bytes,
        accepted,
        CpuReplayMode::Expected(expected_cpu).into(),
        inputs,
        budget,
        true,
        join,
    )
}

fn validate_using<Failure>(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    origins: ReplaySelection<'_>,
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
    original_history: bool,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<Output, Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    let origin_header = origins.working_header();
    let header = header::<Failure>(std::mem::size_of_val(&join))?
        .checked_add(origin_header)
        .ok_or(Resource::Arithmetic)?;
    account::transfer_using(budget, |budget| {
        origins.require_backing(budget)?;
        require_backing(packet_bytes, &inputs, origins.backing_bytes(), budget)?;
        account::temporary_using(budget, header, |budget| {
            require_limits(
                inputs.decoded_history.frame().limits(),
                inputs.expected_limits,
                budget,
            )?;
            let history = if original_history {
                inputs
                    .decoded_history
                    .check_semantics_in_original_account_v1(budget)
            } else {
                inputs.decoded_history.check_semantics(budget)
            }
            .map_err(|error| Error(Cause::History(error)))?;
            let storage = history.storage().retained_storage();
            account::temporary_using(budget, storage, move |budget| {
                let result = with_decoded_native_conditional_source_packet_v2(
                    packet_bytes,
                    budget,
                    |packet, budget| {
                        origins.require_roster(packet.roots, accepted, budget)?;
                        reconstruct::reconstruct_using(
                            packet,
                            accepted,
                            budget,
                            |source, packet, roots, retained, budget| {
                                origins.with_projection(
                                    source,
                                    packet,
                                    budget,
                                    |origins, budget| {
                                        check_source(
                                            source, packet, accepted, origins, roots, retained,
                                            &inputs, &history, budget, join,
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
                .map_err(|error| Error(Cause::Packet(error)));
                drop(history);
                result.map_err(Failure::from)?
            })
        })
    })
}

fn require_limits(actual: Limits, expected: Limits, budget: &mut Budget<'_>) -> Result<(), Error> {
    budget.charge_work(size_of::<Limits>() + 1)?;
    if actual != expected {
        return Err(Error::mismatch(
            "independently accepted complete history limits",
        ));
    }
    Ok(())
}

fn require_backing(
    bytes: &[u8],
    inputs: &Inputs<'_, '_, '_>,
    origin_backing: usize,
    budget: &mut Budget<'_>,
) -> Result<(), Error> {
    budget.charge_work(10)?;
    // Visible minimum only. Caller retains actual capacities/receipts, including
    // separate published bytes when they do not borrow the already-paid F/wire.
    let lengths = [
        bytes.len(),
        inputs.decoded_history.storage().retained_storage(),
        inputs.decoded_history.frame().storage().retained_storage(),
        inputs.decoded_history.frame().canonical_bytes().len(),
        DESCRIPTOR_TABLE_VIEW_STORAGE_V5,
        inputs.descriptors.canonical_bytes().len(),
        size_of::<Catalog>(),
        inputs.final_catalog.canonical_bytes().len(),
        inputs.final_llvm.len(),
    ];
    let minimum = lengths
        .into_iter()
        .try_fold(origin_backing, |sum, n| sum.checked_add(n))
        .ok_or(Resource::Arithmetic)?;
    if budget.storage() < minimum {
        return Err(Resource::Accounting.into());
    }
    Ok(())
}

fn check_source<Failure>(
    source: &ReplayedNativeSourceV1,
    packet: &NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    origins: ProjectedSelection<'_>,
    roots: &mut Vec<ReplayedRoot>,
    retained: &mut usize,
    inputs: &Inputs<'_, '_, '_>,
    history: &History<'_>,
    budget: &mut Budget<'_>,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<(), Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    let (coordinates, storage) = check_production_target_coordinate_preservation_v1(
        source.source().executable(),
        inputs.decoded_history.graph(Role::B),
        inputs.profile,
        budget,
    )
    .map_err(|error| Error(Cause::Coordinates(error)))?;
    account::temporary_using(budget, storage.retained_storage(), move |budget| {
        roster::check(source, packet, history.output(), inputs.descriptors, budget)?;
        reconstruct::reconstruct_roots(
            source,
            packet,
            accepted,
            roots,
            retained,
            budget,
            |ordinal, source, row, policy, body, budget| {
                let origin = origins.at(ordinal, row.semantic_root, budget)?;
                root::reconstruct_root_using(
                    source,
                    row,
                    policy,
                    origin,
                    budget,
                    |request, decoded, budget| {
                        let check = |execution: &crate::ProductionConditionalFormulaExecutionV2,
                                     budget: &mut Budget<'_>| {
                            request
                                .check_replayed_refined_forwarding_output_v1(
                                    &coordinates,
                                    history,
                                    inputs.expected_limits,
                                    budget,
                                )
                                .map_err(|error| Error(Cause::ConditionalFinal(error)))?;
                            check_conditional_contract_content_v2(
                                request,
                                execution,
                                (SemanticFunctionIdV1::from_index(row.semantic_root), body),
                                KernelId::from_bytes(row.launch.kernel_binding()),
                                inputs.descriptors,
                                None,
                                budget,
                            )
                            .map_err(|error| Error(Cause::Contract(error)))
                        };
                        match decoded {
                            DecodedCpu::Registration(decoded) => {
                                import_and_check_conditional_ranked_formula_v2(
                                    request,
                                    decoded,
                                    row.formula_receipt,
                                    policy.formula,
                                    budget,
                                    check,
                                )
                            }
                            DecodedCpu::Policy(decoded) => {
                                import_and_check_conditional_ranked_formula_policy_v2(
                                    request,
                                    decoded,
                                    row.formula_receipt,
                                    policy.formula,
                                    budget,
                                    check,
                                )
                            }
                        }
                    },
                    |imported| imported.map_err(|error| Error(Cause::Import(error)))?,
                )
            },
        )?;
        let relation = check_native_v12_text_descriptor_relation_v5(
            history.output(),
            inputs.final_catalog,
            inputs.published_output_bytes,
            inputs.profile,
            inputs.descriptors,
            inputs.final_llvm,
            budget,
        )
        .map_err(|error| Error(Cause::Text(error)))?;
        account::temporary_using(
            budget,
            relation.storage().retained_storage(),
            move |budget| {
                let result = join(source, inputs.decoded_history, &relation, budget);
                drop(relation);
                result
            },
        )?;
        drop(coordinates);
        Ok(())
    })
}

#[cfg(test)]
#[path = "compiler_native_conditional_final_v2/tests.rs"]
mod tests;

/// Mapped counterpart over a caller-owned inert inventory. Original capture and
/// admission are external duties; native recovery binds the actual capsule field.
pub fn validate_native_conditional_source_through_f_with_cpu_mapping_v2(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    mapping: &NativeConditionalCpuMappingContextV1<'_>,
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
) -> Result<Output, Error> {
    validate_using(
        packet_bytes,
        accepted,
        ReplaySelection::Mapping(mapping),
        inputs,
        budget,
        false,
        |_, _, _, _| Ok(()),
    )
}

pub(crate) fn validate_native_conditional_source_through_f_with_cpu_mapping_using_v2<Failure>(
    packet_bytes: &[u8],
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    mapping: &NativeConditionalCpuMappingContextV1<'_>,
    inputs: Inputs<'_, '_, '_>,
    budget: &mut Budget<'_>,
    original_history: bool,
    join: impl FnOnce(
        &ReplayedNativeSourceV1,
        &DecodedHistory<'_, '_>,
        &Relation<'_, '_, '_, '_, '_>,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<Output, Failure>
where
    Failure: From<Error> + From<SourceError> + From<Resource>,
{
    validate_using(
        packet_bytes,
        accepted,
        ReplaySelection::Mapping(mapping),
        inputs,
        budget,
        original_history,
        join,
    )
}
