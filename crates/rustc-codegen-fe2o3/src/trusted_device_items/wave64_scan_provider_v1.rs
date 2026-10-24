//! Exact reviewed scan implementation, still traversed as ordinary MIR.

use super::*;
use rustc_hir::{Mutability, def::DefKind};
use rustc_middle::ty::{ClauseKind, GenericArgKind, GenericParamDefKind};

const MARKER: &str = "fe2o3_device_gfx942_wave64_inclusive_scan_helper_v1";
const PATH: &str = "fe2o3_device::collective::wave64_inclusive_scan";

fn same_provider(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    expected_path: &str,
    origin: &ReviewedProviderSemanticDefinitionV1,
) -> Result<(), String> {
    let observed = reviewed_provider_semantic_definition_v1(tcx, def_id)?;
    validate_safe_execution_provider_definition_v1(&observed)?;
    if observed.canonical_definition_path != expected_path
        || observed.provider != origin.provider
        || observed.cargo_metadata_build_observation != origin.cargo_metadata_build_observation
        || observed.source_closure_identity != origin.source_closure_identity
    {
        return Err("Wave64 scan helper crosses its reviewed provider".into());
    }
    Ok(())
}

pub(super) fn validate_definition(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    origin: &ReviewedProviderSemanticDefinitionV1,
) -> Result<(), String> {
    let generics = tcx.generics_of(def_id);
    if def_id.is_local()
        || tcx.def_kind(def_id) != DefKind::Fn
        || tcx.get_diagnostic_item(Symbol::intern(MARKER)) != Some(def_id)
        || generics.parent.is_some()
        || generics.parent_count != 0
        || generics.own_params.len() != 1
        || generics.own_params[0].index != 0
        || !matches!(
            generics.own_params[0].kind,
            GenericParamDefKind::Type { .. }
        )
    {
        return Err("Wave64 scan helper is not its exact generic diagnostic definition".into());
    }
    same_provider(tcx, def_id, PATH, origin)?;
    // The primitive definition authenticates its actual implementation, trait
    // item and complete sealed supertrait chain. Join this helper's declared
    // bound to that same actual trait rather than accepting its display name.
    let primitive = definition(
        tcx,
        TrustedDeviceItem::Gfx942Wave64Shuffle(Wave64ShuffleScalarV1::U32),
    )
    .ok_or("Wave64 scan helper lacks the reviewed scalar primitive")?;
    let implementation = tcx
        .impl_of_assoc(primitive)
        .ok_or("Wave64 scan primitive lacks its actual impl")?;
    let trait_id = tcx.impl_trait_id(implementation);
    same_provider(
        tcx,
        trait_id,
        "fe2o3_device::collective::Gfx942CollectiveElement",
        origin,
    )?;
    let mut bound_count = 0;
    for (clause, _) in tcx.predicates_of(def_id).predicates {
        let ClauseKind::Trait(predicate) = clause.kind().skip_binder() else {
            continue;
        };
        if predicate.trait_ref.def_id != trait_id {
            continue;
        }
        if predicate.trait_ref.args.len() != 1
            || !matches!(predicate.trait_ref.self_ty().kind(), TyKind::Param(parameter) if parameter.index == 0)
        {
            return Err("Wave64 scan helper has a different sealed scalar bound".into());
        }
        bound_count += 1;
    }
    if bound_count != 1 {
        return Err("Wave64 scan helper lacks its unique sealed scalar bound".into());
    }
    let signature =
        tcx.instantiate_bound_regions_with_erased(tcx.fn_sig(def_id).instantiate_identity());
    if signature.safety != Safety::Unsafe
        || signature.abi != ExternAbi::Rust
        || signature.c_variadic
        || signature.inputs().len() != 3
        || !matches!(signature.inputs()[0].kind(), TyKind::Uint(UintTy::U32))
        || !matches!(signature.inputs()[2].kind(), TyKind::Param(parameter) if parameter.index == 0)
        || signature.inputs()[2] != signature.output()
    {
        return Err("Wave64 scan helper has a different unsafe Rust signature".into());
    }
    let TyKind::Ref(_, context, Mutability::Not) = signature.inputs()[1].kind() else {
        return Err("Wave64 scan helper requires its shared context reference".into());
    };
    let TyKind::Adt(context, arguments) = context.kind() else {
        return Err("Wave64 scan helper context is not nominal".into());
    };
    if !arguments.is_empty()
        || definition(tcx, TrustedDeviceItem::Gfx942CollectivesContext) != Some(context.did())
    {
        return Err("Wave64 scan helper context identity differs".into());
    }
    same_provider(
        tcx,
        context.did(),
        "fe2o3_device::collective::Gfx942Collectives",
        origin,
    )
}

pub(crate) fn is_authenticated_gfx942_wave64_scan_instance_v1(
    tcx: TyCtxt<'_>,
    instance: Instance<'_>,
    expected_target: &str,
) -> bool {
    if !matches!(expected_target, "gfx942:xnack-" | "gfx950:xnack-")
        || !matches!(instance.def, InstanceKind::Item(_))
        || instance.args.len() != 1
    {
        return false;
    }
    let GenericArgKind::Type(scalar) = instance.args[0].kind() else {
        return false;
    };
    [
        Wave64ShuffleScalarV1::U32,
        Wave64ShuffleScalarV1::I32,
        Wave64ShuffleScalarV1::F32,
    ]
    .into_iter()
    .any(|candidate| candidate.matches(scalar))
        && classify(tcx, instance.def_id())
            == Some(TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper)
}

#[cfg(test)]
#[path = "wave64_scan_provider_v1_tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) use tests::check_actual_scan_instances_v1;
