//! Exact reviewed primitive identity; no generic unsafe-helper permission.

use super::*;
use rustc_hir::{Mutability, def::DefKind};
use rustc_middle::ty::{ClauseKind, IntTy};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Wave64ShuffleScalarV1 {
    U32,
    I32,
    F32,
}

impl Wave64ShuffleScalarV1 {
    pub(crate) const fn marker(self) -> &'static str {
        match self {
            Self::U32 => "fe2o3_device_gfx942_wave64_shuffle_u32_v1",
            Self::I32 => "fe2o3_device_gfx942_wave64_shuffle_i32_v1",
            Self::F32 => "fe2o3_device_gfx942_wave64_shuffle_f32_v1",
        }
    }

    pub(crate) const fn terminal_tag(self) -> u8 {
        match self {
            Self::U32 => 135,
            Self::I32 => 136,
            Self::F32 => 137,
        }
    }

    pub(crate) fn matches(self, ty: Ty<'_>) -> bool {
        matches!(
            (self, ty.kind()),
            (Self::U32, TyKind::Uint(UintTy::U32))
                | (Self::I32, TyKind::Int(IntTy::I32))
                | (Self::F32, TyKind::Float(FloatTy::F32))
        )
    }
}

fn same_reviewed_definition(
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
        return Err("Wave64 primitive crosses its exact reviewed provider definition".into());
    }
    Ok(())
}

fn exact_supertrait(
    tcx: TyCtxt<'_>,
    trait_id: DefId,
    expected_path: &str,
    origin: &ReviewedProviderSemanticDefinitionV1,
) -> Result<DefId, String> {
    let mut found = None;
    for (predicate, _) in tcx
        .explicit_super_predicates_of(trait_id)
        .iter_identity_copied()
    {
        let ClauseKind::Trait(predicate) = predicate.kind().skip_binder() else {
            continue;
        };
        let candidate = predicate.trait_ref.def_id;
        if candidate.krate != trait_id.krate {
            continue;
        }
        // Match the same structural identity authenticated by the provider;
        // def_path_str is a display path and can select a public re-export.
        let structural = tcx.def_path(candidate).to_string_no_crate_verbose();
        let canonical = canonical_compiler_definition_path(
            tcx.crate_name(candidate.krate).as_str(),
            &structural,
        )?;
        if canonical != expected_path {
            continue;
        }
        if !matches!(predicate.trait_ref.self_ty().kind(), TyKind::Param(parameter) if parameter.index == 0)
            || found.replace(candidate).is_some()
        {
            return Err("Wave64 primitive has an ambiguous sealed-trait chain".into());
        }
        same_reviewed_definition(tcx, candidate, expected_path, origin)?;
    }
    found.ok_or_else(|| "Wave64 primitive lacks its exact sealed-trait chain".into())
}

pub(super) fn validate_definition(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    scalar: Wave64ShuffleScalarV1,
    origin: &ReviewedProviderSemanticDefinitionV1,
) -> Result<(), String> {
    if def_id.is_local()
        || tcx.get_diagnostic_item(Symbol::intern(scalar.marker())) != Some(def_id)
        || tcx.generics_of(def_id).count() != 0
    {
        return Err("Wave64 primitive is not its exact monomorphic diagnostic definition".into());
    }
    let associated = tcx
        .opt_associated_item(def_id)
        .ok_or_else(|| "Wave64 primitive is not an associated implementation".to_owned())?;
    let implementation = tcx
        .impl_of_assoc(def_id)
        .ok_or_else(|| "Wave64 primitive lacks its actual implementation".to_owned())?;
    if !associated.is_fn()
        || associated.is_method()
        || associated.impl_container(tcx) != Some(implementation)
        || !tcx.impl_is_of_trait(implementation)
        || !scalar.matches(tcx.type_of(implementation).instantiate_identity())
    {
        return Err("Wave64 primitive has a different associated ABI or scalar Self".into());
    }
    let trait_id = tcx.impl_trait_id(implementation);
    let trait_item = associated
        .trait_item_def_id()
        .ok_or_else(|| "Wave64 primitive lacks its actual trait-item binding".to_owned())?;
    if tcx.parent(trait_item) != trait_id
        || tcx.def_kind(trait_id) != DefKind::Trait
        || tcx.item_name(def_id).as_str() != "__fe2o3_wave64_shuffle_index"
        || tcx.item_name(trait_item) != tcx.item_name(def_id)
    {
        return Err("Wave64 primitive has a different trait-item binding".into());
    }
    same_reviewed_definition(
        tcx,
        trait_id,
        "fe2o3_device::collective::Gfx942CollectiveElement",
        origin,
    )?;
    let workgroup = exact_supertrait(
        tcx,
        trait_id,
        "fe2o3_device::collective::WorkgroupCollectiveElement",
        origin,
    )?;
    exact_supertrait(
        tcx,
        workgroup,
        "fe2o3_device::collective::sealed::CollectiveElement",
        origin,
    )?;
    let signature =
        tcx.instantiate_bound_regions_with_erased(tcx.fn_sig(def_id).instantiate_identity());
    if signature.safety != Safety::Unsafe
        || signature.abi != ExternAbi::Rust
        || signature.c_variadic
        || signature.inputs().len() != 3
        || !scalar.matches(signature.inputs()[1])
        || signature.inputs()[1] != signature.output()
        || !matches!(signature.inputs()[2].kind(), TyKind::Uint(UintTy::U32))
    {
        return Err("Wave64 primitive has a different unsafe Rust scalar signature".into());
    }
    let TyKind::Ref(_, context, Mutability::Not) = signature.inputs()[0].kind() else {
        return Err("Wave64 primitive requires its shared context reference".into());
    };
    let TyKind::Adt(context, arguments) = context.kind() else {
        return Err("Wave64 primitive context is not the reviewed nominal type".into());
    };
    if !arguments.is_empty()
        || definition(tcx, TrustedDeviceItem::Gfx942CollectivesContext) != Some(context.did())
    {
        return Err("Wave64 primitive context identity differs".into());
    }
    same_reviewed_definition(
        tcx,
        context.did(),
        "fe2o3_device::collective::Gfx942Collectives",
        origin,
    )
}

pub(crate) fn scalar_for_instance(
    tcx: TyCtxt<'_>,
    instance: Instance<'_>,
) -> Option<Wave64ShuffleScalarV1> {
    if !instance.args.is_empty() || !matches!(instance.def, InstanceKind::Item(_)) {
        return None;
    }
    match classify(tcx, instance.def_id())? {
        TrustedDeviceItem::Gfx942Wave64Shuffle(scalar) => Some(scalar),
        _ => None,
    }
}

#[cfg(test)]
#[path = "wave64_shuffle_actual_paths_v1_tests.rs"]
mod actual_paths;
#[cfg(test)]
pub(crate) use actual_paths::check_actual_sealed_trait_chain_paths_v1;

// The only target check belongs to a real collector target, not the semantic
// layout carrier. Wire capture itself is never evidence of full participation.
pub(crate) fn is_authenticated_gfx942_wave64_shuffle_instance_v1(
    tcx: TyCtxt<'_>,
    instance: Instance<'_>,
    expected_target: &str,
) -> bool {
    // Authenticate traversal of the same provider implementation on both Wave64
    // profiles. This does not materialize participation, lane identity or epochs:
    // the later raw-source shuffle materializer still refuses that missing join.
    matches!(expected_target, "gfx942:xnack-" | "gfx950:xnack-")
        && scalar_for_instance(tcx, instance).is_some()
}
