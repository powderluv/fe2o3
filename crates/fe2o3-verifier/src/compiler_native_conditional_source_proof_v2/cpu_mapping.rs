//! Inert projection of independently selected inventory coordinates onto real source.
use super::cpu_origin::{
    CpuReplayMode, CpuSubjects, NativeConditionalCpuOriginExpectationV1 as Origin,
};
use super::*;
use crate::portable_reference_v1::codec::{
    MAX_REFERENCE_ENROLLMENT_BINDINGS_V1, ReferenceEnrollmentOriginV1,
};
use fe2o3_compiler_lineage::{RustcEnrollmentInventoryRefV1, RustcEnrollmentInventoryRootV1};
use fe2o3_mir_model::semantic_mir_v1::SemanticFunctionRoleV1;
use fe2o3_mir_model::semantic_mir_v1::{SemanticFunctionDeclV1, SemanticFunctionIdV1};
use sha2::{Digest, Sha256};

pub use fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1;

const _: () = assert!(
    MAX_REFERENCE_ENROLLMENT_BINDINGS_V1 as usize
        == fe2o3_rustc_invocation::MAX_REFERENCE_ENROLLMENT_BINDINGS_V1
);

/// Borrowed inert content for the lower source/final consumers. Caller prepays
/// actual backing capacity, this view and its independent expectation. The lower
/// backing check sees only the wire byte length, not its owner's allocation
/// capacity. Native recovery retains the actual handoff capacity and decodes its
/// retained capsule inventory internally.
///
/// ```compile_fail
/// use fe2o3_compiler_lineage::RustcEnrollmentInventoryRefV1;
/// use fe2o3_verifier::{NativeConditionalCpuMappingContextV1 as Context, NativeConditionalCpuMappingExpectationV1 as Expected};
/// fn escape<'a>(inventory: RustcEnrollmentInventoryRefV1<'a>, expected: &'a Expected) -> Context<'a> {
///     Context { inventory: &inventory, expected }
/// }
/// ```
pub struct NativeConditionalCpuMappingContextV1<'a> {
    pub inventory: &'a RustcEnrollmentInventoryRefV1<'a>,
    pub expected: &'a NativeConditionalCpuMappingExpectationV1,
}

#[derive(Clone, Copy)]
pub(super) enum ReplaySelection<'a> {
    Legacy(CpuReplayMode<'a>),
    Mapping(&'a NativeConditionalCpuMappingContextV1<'a>),
}
impl<'a> From<CpuReplayMode<'a>> for ReplaySelection<'a> {
    fn from(value: CpuReplayMode<'a>) -> Self {
        Self::Legacy(value)
    }
}

#[derive(Clone, Copy)]
pub(super) struct ProjectedRoot {
    semantic_root: u32,
    origin: Origin,
    kernel: [u8; 32],
    reference: [u8; 32],
}
#[derive(Clone, Copy)]
pub(super) struct RootExpectation<'a> {
    pub origin: Origin,
    functions: Option<&'a ProjectedRoot>,
}
impl From<Origin> for RootExpectation<'_> {
    fn from(origin: Origin) -> Self {
        Self {
            origin,
            functions: None,
        }
    }
}
impl RootExpectation<'_> {
    pub(super) fn require_subjects(
        self,
        cpu: &CpuSubjects<'_>,
        budget: &mut Budget<'_>,
    ) -> Result<(), E> {
        if let Some(row) = self.functions {
            budget.charge_work(32 + 32 + 2)?;
            if cpu.kernel.function_sha256 != row.kernel
                || cpu.reference.function_sha256 != row.reference
            {
                return Err(E::invalid(
                    "independent mapped CPU kernel/reference identities",
                ));
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub(super) enum ProjectedSelection<'a> {
    Legacy(CpuReplayMode<'a>),
    Mapped(&'a [ProjectedRoot]),
}
impl<'a> ProjectedSelection<'a> {
    pub(super) fn at(
        self,
        ordinal: usize,
        semantic_root: u32,
        budget: &mut Budget<'_>,
    ) -> Result<RootExpectation<'a>, E> {
        match self {
            Self::Legacy(mode) => Ok(mode.at(ordinal, semantic_root, budget)?.into()),
            Self::Mapped(rows) => {
                budget.charge_work(2)?;
                let row = rows
                    .get(ordinal)
                    .ok_or_else(|| E::invalid("missing projected CPU mapping"))?;
                if row.semantic_root != semantic_root {
                    return Err(E::invalid("changed projected CPU root order"));
                }
                Ok(RootExpectation {
                    origin: row.origin,
                    functions: Some(row),
                })
            }
        }
    }
}

impl ReplaySelection<'_> {
    pub(super) fn backing_bytes(self) -> usize {
        match self {
            Self::Legacy(mode) => mode.backing_bytes(),
            Self::Mapping(context) => {
                // Borrowed bytes expose a visible floor, not allocation capacity.
                context.inventory.canonical_bytes().len()
                    + size_of::<RustcEnrollmentInventoryRefV1<'_>>()
                    + size_of::<NativeConditionalCpuMappingContextV1<'_>>()
                    + size_of::<NativeConditionalCpuMappingExpectationV1>()
            }
        }
    }
    pub(super) fn working_header(self) -> usize {
        match self {
            Self::Legacy(CpuReplayMode::RegistrationOnly) => 0,
            Self::Legacy(CpuReplayMode::Expected(_)) => size_of::<CpuReplayMode<'_>>(),
            Self::Mapping(_) => size_of::<Self>(),
        }
    }
    pub(super) fn require_backing(self, budget: &mut Budget<'_>) -> Result<(), E> {
        match self {
            Self::Legacy(mode) => mode.require_backing(budget),
            Self::Mapping(_) => {
                budget.charge_work(1)?;
                if budget.storage() < self.backing_bytes() {
                    return Err(Resource::Accounting.into());
                }
                Ok(())
            }
        }
    }
    pub(super) fn require_roster(
        self,
        roots: &[NativeConditionalSourceRootV2<'_>],
        accepted: &[NativeConditionalRootPolicyV2<'_>],
        budget: &mut Budget<'_>,
    ) -> Result<(), E> {
        match self {
            Self::Legacy(mode) => mode.require_roster(roots, accepted, budget),
            Self::Mapping(context) => {
                require_header(context, budget)?;
                budget.charge_work(3)?;
                if roots.is_empty()
                    || roots.len() != accepted.len()
                    || roots.len() != context.inventory.root_count()
                {
                    return Err(E::invalid("complete mapped CPU root/policy roster"));
                }
                Ok(())
            }
        }
    }

    pub(super) fn with_projection<R, Failure: From<E> + From<Resource>>(
        self,
        source: &ReplayedNativeSourceV1,
        packet: &NativeConditionalSourcePacketInputV2<'_>,
        budget: &mut Budget<'_>,
        consume: impl FnOnce(ProjectedSelection<'_>, &mut Budget<'_>) -> Result<R, Failure>,
    ) -> Result<R, Failure> {
        let Self::Mapping(context) = self else {
            let Self::Legacy(mode) = self else {
                unreachable!()
            };
            return consume(ProjectedSelection::Legacy(mode), budget);
        };
        projected_scope(
            packet.roots.len(),
            budget,
            |projected, budget| project(context, source, packet, projected, budget),
            consume,
        )
    }
}

fn projected_scope<R, Failure: From<E> + From<Resource>>(
    count: usize,
    budget: &mut Budget<'_>,
    build: impl FnOnce(&mut Vec<ProjectedRoot>, &mut Budget<'_>) -> Result<(), E>,
    consume: impl FnOnce(ProjectedSelection<'_>, &mut Budget<'_>) -> Result<R, Failure>,
) -> Result<R, Failure> {
    budget.check_prior_denials_v1()?;
    let header = [
        size_of::<Vec<ProjectedRoot>>(),
        size_of::<ProjectedSelection<'_>>(),
        size_of::<ReplaySelection<'_>>(),
        size_of::<RootExpectation<'_>>(),
        2 * size_of::<RustcEnrollmentInventoryRootV1>(),
        size_of::<Option<RustcEnrollmentInventoryRootV1>>(),
        size_of::<Sha256>(),
        256,
        6 * size_of::<usize>(),
        size_of::<Result<R, Failure>>(),
        std::mem::size_of_val(&build),
        std::mem::size_of_val(&consume),
    ]
    .into_iter()
    .try_fold(0usize, |n, v| n.checked_add(v))
    .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(header)?;
    let (mut projected, capacity) = account::vector(count, budget)?;
    let paid = header.checked_add(capacity).ok_or(Resource::Arithmetic)?;
    build(&mut projected, budget)?;
    let floor = budget.storage();
    let ledger = budget.work_ledger_identity_v1();
    let storage_account = budget.storage_account_identity_v1();
    let address = budget as *mut Budget<'_>;
    let result = consume(ProjectedSelection::Mapped(&projected), budget);
    drop(projected);
    if budget as *mut Budget<'_> != address
        || budget.work_ledger_identity_v1() != ledger
        || budget.storage_account_identity_v1() != storage_account
        || budget.storage() < floor
    {
        drop(result);
        return Err(Resource::Accounting.into());
    }
    if let Err(error) = budget.check_prior_denials_v1() {
        drop(result);
        return Err(error.into());
    }
    match result {
        Ok(value) => {
            budget.release_storage(paid)?;
            Ok(value)
        }
        Err(error) => Err(error),
    }
}

fn require_header(
    context: &NativeConditionalCpuMappingContextV1<'_>,
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    budget.charge_work(NativeConditionalCpuMappingExpectationV1::HEADER_MATCH_WORK)?;
    let header = context.inventory.header();
    let expected = context.expected;
    if !expected.matches_header(&header) {
        return Err(E::invalid(
            "independent mapped CPU invocation/policy coordinates",
        ));
    }
    Ok(())
}

fn copy_root(
    row: fe2o3_compiler_lineage::RustcEnrollmentInventoryRootRefV1<'_>,
    budget: &mut Budget<'_>,
) -> Result<RustcEnrollmentInventoryRootV1, E> {
    budget.charge_work(144 + size_of::<RustcEnrollmentInventoryRootV1>())?;
    Ok(row.value())
}

fn project(
    context: &NativeConditionalCpuMappingContextV1<'_>,
    source: &ReplayedNativeSourceV1,
    packet: &NativeConditionalSourcePacketInputV2<'_>,
    projected: &mut Vec<ProjectedRoot>,
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    let semantic = source.source().semantic_ssa().source_semantic();
    project_components(
        context,
        semantic.functions(),
        semantic.roots(),
        packet,
        projected,
        budget,
    )
}

fn project_components(
    context: &NativeConditionalCpuMappingContextV1<'_>,
    functions: &[SemanticFunctionDeclV1],
    source_roots: &[SemanticFunctionIdV1],
    packet: &NativeConditionalSourcePacketInputV2<'_>,
    projected: &mut Vec<ProjectedRoot>,
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    let mut inventory = context.inventory.roots();
    // Match the complete canonical KernelEntry subset, not a caller-selected subset.
    for (index, function) in functions.iter().enumerate() {
        budget.charge_work(1)?;
        let Some(entry) = function.kernel_entry() else {
            continue;
        };
        let row = copy_root(
            inventory
                .next()
                .ok_or_else(|| E::invalid("missing canonical kernel mapping"))?,
            budget,
        )?;
        budget.charge_work(32 + 32 + 6)?;
        if row.semantic_root as usize != index
            || function.role() != SemanticFunctionRoleV1::KernelRoot
            || row.kernel_instance != *function.identity().as_bytes()
            || row.kernel_binding != *entry.kernel_binding_identity().as_bytes()
        {
            return Err(E::invalid(
                "mapped actual semantic kernel identity/role/binding",
            ));
        }
    }
    budget.charge_work(1)?;
    if inventory.next().is_some() {
        return Err(E::invalid("extra canonical kernel mapping"));
    }
    for (ordinal, root) in packet.roots.iter().enumerate() {
        budget.charge_work(2)?;
        if source_roots.get(ordinal).map(|id| id.index()) != Some(root.semantic_root) {
            return Err(E::invalid("mapped actual semantic root order"));
        }
        let mut selected = None;
        for view in context.inventory.roots() {
            let row = copy_root(view, budget)?;
            if row.semantic_root == root.semantic_root {
                if selected.replace(row).is_some() {
                    return Err(E::invalid("duplicate mapped semantic root"));
                }
            }
        }
        let row = selected.ok_or_else(|| E::invalid("missing mapped semantic root"))?;
        for previous in projected.iter() {
            budget.charge_work(1)?;
            if previous.semantic_root == row.semantic_root {
                return Err(E::invalid("duplicate projected semantic root"));
            }
        }
        budget.charge_work(
            root.launch
                .logical_name()
                .len()
                .checked_add(128 + 32 + 32 + 5)
                .ok_or(Resource::Arithmetic)?,
        )?;
        let name: [u8; 32] = Sha256::digest(root.launch.logical_name().as_bytes()).into();
        if usize::try_from(row.logical_name_len).map_err(|_| Resource::Arithmetic)?
            != root.launch.logical_name().len()
            || row.logical_name_sha256 != name
            || row.kernel_binding != root.launch.kernel_binding()
        {
            return Err(E::invalid("mapped actual logical name/kernel binding"));
        }
        let origin = match row.origin_tag {
            0 if row.descriptor_ordinal == 0 => Origin::SourceRegistrationV1,
            1 if row.descriptor_ordinal < context.expected.enrollment_binding_count => {
                Origin::ReferenceEnrollmentV1(ReferenceEnrollmentOriginV1 {
                    rustc_invocation_sha256: context.expected.rustc_invocation_sha256,
                    native_policy_sha256: context.expected.native_policy_sha256,
                    policy_generation: context.expected.policy_generation,
                    mapping_ordinal: row.descriptor_ordinal,
                })
            }
            _ => return Err(E::invalid("mapped CPU origin tag/ordinal")),
        };
        projected.push(ProjectedRoot {
            semantic_root: row.semantic_root,
            origin,
            kernel: row.kernel_instance,
            reference: row.reference_instance,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests/cpu_mapping.rs"]
mod tests;
