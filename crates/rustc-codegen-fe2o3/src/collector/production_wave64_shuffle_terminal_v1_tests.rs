use super::*;
use crate::trusted_device_items::{TrustedDeviceItem, definition};
use fe2o3_mir_model::semantic_mir_v1::SemanticTypeIdentityV1;
use rustc_hir::def::DefKind;

// Called only by genuine source callbacks with the production-imported table.
pub(crate) fn check_wave64_descriptor_mutations_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    scalar: Wave64ShuffleScalarV1,
    abi: &SemanticFunctionAbiV1,
    types: &[SemanticTypeDeclV1],
) {
    let exact = construct(tcx, instance, scalar, abi, types).expect("actual imported primitive");
    let SemanticCompilerIntrinsicOperationV1::Gfx942Wave64ShuffleIndex { context, element } = exact
    else {
        unreachable!()
    };
    for other in [
        Wave64ShuffleScalarV1::U32,
        Wave64ShuffleScalarV1::I32,
        Wave64ShuffleScalarV1::F32,
    ] {
        if other != scalar {
            assert!(construct(tcx, instance, other, abi, types).is_none());
        }
    }
    for target in [
        "gfx942",
        "gfx942:xnack+",
        "gfx950",
        "gfx950:xnack+",
        "gfx950:xnack- ",
        "gfx950:xnack-:wave32",
        "gfx950:wave32:xnack-",
        "gfx90a:xnack-",
        "",
        "gfx942:xnack- ",
    ] {
        assert!(
            !trusted_device_items::is_authenticated_gfx942_wave64_shuffle_instance_v1(
                tcx, instance, target,
            )
        );
    }
    for target in ["gfx942:xnack-", "gfx950:xnack-"] {
        assert!(
            trusted_device_items::is_authenticated_gfx942_wave64_shuffle_instance_v1(
                tcx, instance, target
            )
        );
    }

    assert_eq!(
        definition(tcx, TrustedDeviceItem::Gfx942Wave64Shuffle(scalar)),
        Some(instance.def_id())
    );
    let name = match scalar {
        Wave64ShuffleScalarV1::U32 => "forged_u32",
        Wave64ShuffleScalarV1::I32 => "forged_i32",
        Wave64ShuffleScalarV1::F32 => "forged_f32",
    };
    let mut forged_functions = tcx
        .hir_body_owners()
        .map(|id| id.to_def_id())
        .filter(|id| tcx.def_kind(*id) == DefKind::Fn && tcx.item_name(*id).as_str() == name);
    let forged = forged_functions
        .next()
        .expect("named local negative function");
    assert!(
        forged_functions.next().is_none(),
        "local negative function must be unique"
    );
    let forged = Instance::mono(tcx, forged);
    assert_eq!(
        source_signature_v1(tcx, instance).unwrap(),
        source_signature_v1(tcx, forged).unwrap(),
        "the local negative really has the provider's source signature"
    );
    assert!(construct(tcx, forged, scalar, abi, types).is_none());
    for target in ["gfx942:xnack-", "gfx950:xnack-"] {
        assert!(
            !trusted_device_items::is_authenticated_gfx942_wave64_shuffle_instance_v1(
                tcx, forged, target
            )
        );
    }

    let with_unwind = |can_unwind| {
        SemanticFunctionAbiV1::from_rustc_with_source_signature(
            abi.identity(),
            abi.layout_identity(),
            abi.canon_abi(),
            abi.extern_abi(),
            can_unwind,
            abi.c_variadic(),
            abi.fixed_count(),
            abi.source_input_types().to_vec(),
            abi.source_output_type(),
            abi.arguments().to_vec(),
            abi.return_value().clone(),
        )
        .unwrap()
        .with_source_argument_ownership(abi.source_argument_ownership().to_vec())
        .unwrap()
    };
    assert_eq!(
        construct(tcx, instance, scalar, &with_unwind(abi.can_unwind()), types),
        Some(exact)
    );
    assert!(
        construct(
            tcx,
            instance,
            scalar,
            &with_unwind(!abi.can_unwind()),
            types
        )
        .is_none()
    );
    let changed = abi
        .clone()
        .with_source_argument_ownership(vec![
            SemanticSourceArgumentOwnershipV1::UniqueBorrow,
            SemanticSourceArgumentOwnershipV1::ByValue,
            SemanticSourceArgumentOwnershipV1::ByValue,
        ])
        .unwrap();
    assert!(construct(tcx, instance, scalar, &changed, types).is_none());
    for id in [
        context,
        element,
        abi.source_input_types()[0],
        abi.source_input_types()[2],
    ] {
        let mut changed = types.to_vec();
        let old = &types[id.index() as usize];
        let mut bytes = [0x5a; 32];
        if old.identity() == SemanticTypeIdentityV1::from_sha256(bytes) {
            bytes[0] ^= 1;
        }
        changed[id.index() as usize] = SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes),
            old.layout_identity(),
            old.layout().clone(),
            old.shape().clone(),
        );
        assert!(
            construct(tcx, instance, scalar, abi, &changed).is_none(),
            "foreign type {id:?}"
        );
    }
    for shape in [
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float { bits: 64 }),
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
            signed: false,
            bits: 64,
        }),
    ] {
        let mut changed = types.to_vec();
        let old = &types[element.index() as usize];
        changed[element.index() as usize] = SemanticTypeDeclV1::new(
            old.identity(),
            old.layout_identity(),
            old.layout().clone(),
            shape,
        );
        assert!(construct(tcx, instance, scalar, abi, &changed).is_none());
    }
}
