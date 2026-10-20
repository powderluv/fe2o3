use super::*;
use fe2o3_compiler_lineage::NativeNeutralModuleRefV1;
use fe2o3_kernel_descriptor::KernelId;
use fe2o3_lower_mir_kernel::{
    ProductionSourceLaunchRootInputV1, replay_native_source_correspondence_v1,
};
use fe2o3_mir_model::{
    InertCanonicalSemanticU32InductionEvidenceV1 as Induction,
    analyze_semantic_u32_induction_no_overflow_v1, semantic_mir_v1::SemanticFunctionIdV1,
};

pub(super) fn roster(
    packet: &NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    budget: &mut Budget<'_>,
) -> Result<(), E> {
    budget.charge_work(4)?;
    if packet.roots.is_empty()
        || packet.roots.len() != accepted.len()
        || packet.canonical_kernel_order.len() != packet.roots.len()
    {
        return Err(E::invalid("complete conditional policy/root/order roster"));
    }
    for (row, policy) in packet.roots.iter().zip(accepted) {
        budget.charge_work(2)?;
        if row.semantic_root != policy.semantic_root
            || row.launch_rank != row.launch.launch().rank()
        {
            return Err(E::invalid("ordered conditional policy/root/launch"));
        }
    }
    // Strict full-binding order plus equal cardinality establishes a permutation;
    // source replay separately checks the binding and semantic-root bijections.
    let mut previous = None;
    for &ordinal in packet.canonical_kernel_order {
        budget.charge_work(33)?;
        let row = packet
            .roots
            .get(usize::try_from(ordinal).map_err(|_| Resource::Arithmetic)?)
            .ok_or_else(|| E::invalid("canonical conditional root ordinal"))?;
        let id = KernelId::from_bytes(row.launch.kernel_binding());
        if previous.is_some_and(|previous| previous >= id) {
            return Err(E::invalid("actual full-binding KernelId order"));
        }
        previous = Some(id);
    }
    Ok(())
}

pub(super) fn reconstruct(
    packet: NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    budget: &mut Budget<'_>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    E,
> {
    reconstruct_with_origins(
        packet,
        accepted,
        cpu_origin::CpuReplayMode::RegistrationOnly,
        budget,
    )
}

pub(super) fn reconstruct_with_origins(
    packet: NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    origins: cpu_origin::CpuReplayMode<'_>,
    budget: &mut Budget<'_>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    E,
> {
    reconstruct_selected(packet, accepted, origins.into(), budget)
}

pub(super) fn reconstruct_selected(
    packet: NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    origins: cpu_mapping::ReplaySelection<'_>,
    budget: &mut Budget<'_>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    E,
> {
    origins.require_roster(packet.roots, accepted, budget)?;
    reconstruct_using(
        packet,
        accepted,
        budget,
        |source, packet, roots, retained, budget| {
            origins.with_projection(source, packet, budget, |origins, budget| {
                reconstruct_roots(
                    source,
                    packet,
                    accepted,
                    roots,
                    retained,
                    budget,
                    |ordinal, source, row, policy, _, budget| {
                        let origin = origins.at(ordinal, row.semantic_root, budget)?;
                        root::reconstruct_root(source, row, policy, origin, budget)
                    },
                )
            })
        },
    )
}

pub(super) fn reconstruct_using<Failure: From<E> + From<Resource>>(
    packet: NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    budget: &mut Budget<'_>,
    visit: impl FnOnce(
        &ReplayedNativeSourceV1,
        &NativeConditionalSourcePacketInputV2<'_>,
        &mut Vec<ReplayedRoot>,
        &mut usize,
        &mut Budget<'_>,
    ) -> Result<(), Failure>,
) -> Result<
    (
        ReplayedNativeConditionalSourceV2,
        NativeConditionalSourceStorageV2,
    ),
    Failure,
> {
    roster(&packet, accepted, budget)?;
    budget.charge_work(packet.native_module.len())?;
    let native = NativeNeutralModuleRefV1::decode(packet.native_module)
        .map_err(|error| E(Cause::Native(error)))?;
    budget.reserve_storage(size_of::<Vec<ProductionSourceLaunchRootInputV1<'_>>>())?;
    let (mut launches, launch_storage) = account::vector(packet.roots.len(), budget)?;
    for row in packet.roots {
        budget.charge_work(1)?;
        launches.push(row.launch);
    }
    let (source, source_storage) = replay_native_source_correspondence_v1(
        packet.semantic_mir,
        native.graph_bytes(),
        native.catalog_bytes(),
        &launches,
        budget,
    )
    .map_err(|error| E(Cause::Source(error)))?;
    budget.reserve_storage(source_storage.retained_storage())?;
    drop(launches);
    budget.release_storage(
        launch_storage
            .checked_add(size_of::<Vec<ProductionSourceLaunchRootInputV1<'_>>>())
            .ok_or(Resource::Arithmetic)?,
    )?;
    let semantic = source.source().semantic_ssa().source_semantic();
    let graph = source.source().executable();
    budget.charge_work(132)?;
    if native.subject().graph_digest() != graph.canonical().identity().digest()
        || native.subject().graph_length()
            != u64::try_from(native.graph_bytes().len()).map_err(|_| Resource::Arithmetic)?
        || native.subject().catalog_digest() != source.catalog().digest()
        || native.subject().catalog_length()
            != u64::try_from(source.catalog().canonical_bytes().len())
                .map_err(|_| Resource::Arithmetic)?
        || packet.roots.len() != semantic.roots().len()
        || packet.roots.len() != source.source().source_launch().roots().len()
        || packet.roots.len() != graph.module().kernels.len()
    {
        return Err(E::invalid("complete reconstructed source/N/catalog roots").into());
    }
    let header = size_of::<ReplayedNativeConditionalSourceV2>()
        .checked_sub(size_of::<ReplayedNativeSourceV1>())
        .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(header)?;
    let (mut roots, root_storage) = account::vector::<ReplayedRoot>(packet.roots.len(), budget)?;
    let (mut order, order_storage) = account::vector(packet.canonical_kernel_order.len(), budget)?;
    for &ordinal in packet.canonical_kernel_order {
        budget.charge_work(1)?;
        order.push(ordinal);
    }
    let mut retained = source_storage
        .retained_storage()
        .checked_add(header)
        .and_then(|n| n.checked_add(root_storage))
        .and_then(|n| n.checked_add(order_storage))
        .ok_or(Resource::Arithmetic)?;
    visit(&source, &packet, &mut roots, &mut retained, budget)?;
    Ok((
        ReplayedNativeConditionalSourceV2 {
            source,
            roots,
            canonical_kernel_order: order,
        },
        NativeConditionalSourceStorageV2(retained),
    ))
}

pub(super) fn reconstruct_roots<Failure: From<E> + From<Resource>>(
    source: &ReplayedNativeSourceV1,
    packet: &NativeConditionalSourcePacketInputV2<'_>,
    accepted: &[NativeConditionalRootPolicyV2<'_>],
    roots: &mut Vec<ReplayedRoot>,
    retained: &mut usize,
    budget: &mut Budget<'_>,
    mut visit: impl FnMut(
        usize,
        &ReplayedNativeSourceV1,
        &NativeConditionalSourceRootV2<'_>,
        &NativeConditionalRootPolicyV2<'_>,
        SemanticFunctionIdV1,
        &mut Budget<'_>,
    ) -> Result<ReplayedRoot, Failure>,
) -> Result<(), Failure> {
    for (ordinal, (row, policy)) in packet.roots.iter().zip(accepted).enumerate() {
        let body = check_source_root(source, ordinal, row, budget)?;
        let root = visit(ordinal, source, row, policy, body, budget)?;
        let payload = root
            .input
            .retained_storage_v1()?
            .checked_add(root.formula.retained_storage_v2())
            .ok_or(Resource::Arithmetic)?
            .checked_sub(size_of::<RetainedProductionConditionalFormulaV2>())
            .ok_or(Resource::Accounting)?;
        roots.push(root);
        // The prepaid root slot now owns the formula's inline receipt header.
        budget.release_storage(size_of::<RetainedProductionConditionalFormulaV2>())?;
        *retained = retained.checked_add(payload).ok_or(Resource::Arithmetic)?;
    }
    Ok(())
}

fn check_source_root(
    source: &ReplayedNativeSourceV1,
    ordinal: usize,
    row: &NativeConditionalSourceRootV2<'_>,
    budget: &mut Budget<'_>,
) -> Result<SemanticFunctionIdV1, E> {
    let semantic = source.source().semantic_ssa().source_semantic();
    let semantic_root = semantic.roots()[ordinal];
    let function = &semantic.functions()[semantic_root.index() as usize];
    let entry = function
        .kernel_entry()
        .ok_or_else(|| E::invalid("semantic root entry"))?;
    let launch = source.source().source_launch().roots()[ordinal];
    budget.charge_work(100)?;
    if row.semantic_root != semantic_root.index()
        || row.launch.kernel_binding() != *entry.kernel_binding_identity().as_bytes()
        || row.launch_rank != launch.source_rank()
        || row
            .launch
            .launch()
            .exact_workgroup()
            .map(|v| v.map(u64::from))
            != Some(launch.layout().workgroup_extents())
    {
        return Err(E::invalid("conditional semantic root/binding/launch"));
    }
    let mut found = false;
    for kernel in &source.source().executable().module().kernels {
        budget.charge_work(
            kernel
                .id
                .as_str()
                .len()
                .checked_add(kernel.entry.as_str().len())
                .and_then(|n| n.checked_add(entry.export_symbol().as_bytes().len()))
                .and_then(|n| n.checked_add(4))
                .ok_or(Resource::Arithmetic)?,
        )?;
        if kernel.id.as_str().as_bytes() == entry.export_symbol().as_bytes() {
            if found
                || kernel.entry.as_str() != kernel.id.as_str()
                || kernel.domain.rank() != row.launch_rank
            {
                return Err(E::invalid("conditional N kernel entry/rank"));
            }
            found = true;
        }
    }
    if !found {
        return Err(E::invalid("missing conditional N kernel"));
    }
    budget.charge_work(
        row.induction_bytes
            .len()
            .checked_add(1)
            .ok_or(Resource::Arithmetic)?,
    )?;
    let selection = semantic
        .select_kernel_body_for_root_v1(semantic_root)
        .ok_or_else(|| E::invalid("selected conditional source body"))?;
    // Inherited semantic induction analysis keeps its existing independent hard
    // limits, as in normal native replay; no fresh canonical ledger is made.
    let report = analyze_semantic_u32_induction_no_overflow_v1(semantic, selection.body())
        .map_err(|error| E(Cause::Induction(error)))?;
    let induction =
        Induction::from_report(&report).map_err(|error| E(Cause::InductionWire(error)))?;
    budget.charge_work(
        induction
            .canonical_bytes()
            .len()
            .checked_add(row.induction_bytes.len())
            .ok_or(Resource::Arithmetic)?,
    )?;
    if induction.canonical_bytes() != row.induction_bytes {
        return Err(E::invalid("exact conditional source induction replay"));
    }
    Ok(selection.body())
}
