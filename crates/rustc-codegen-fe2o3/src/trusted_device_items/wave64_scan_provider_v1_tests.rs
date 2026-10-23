//! Called only from genuine source callbacks with the actual rustc owner.
use super::*;

pub(crate) fn check_actual_scan_instances_v1(tcx: TyCtxt<'_>) {
    let helper = definition(tcx, TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper)
        .expect("reviewed scan helper diagnostic identity");
    let origin = reviewed_provider_semantic_definition_v1(tcx, helper).unwrap();
    validate_definition(tcx, helper, &origin).unwrap();
    for target in ["gfx942:xnack-", "gfx950:xnack-"] {
        for scalar in [tcx.types.u32, tcx.types.i32, tcx.types.f32] {
            let instance = Instance::new_raw(helper, tcx.mk_args(&[scalar.into()]));
            assert!(is_authenticated_gfx942_wave64_scan_instance_v1(
                tcx, instance, target
            ));
            for target in [
                "gfx942",
                "gfx950",
                "gfx950:xnack+",
                "gfx950:xnack- ",
                "gfx950:xnack-:wave32",
                "gfx950:wave32:xnack-",
                "gfx950:xnack-:wave64",
                "gfx942:xnack+",
                "",
                "gfx90a:xnack-",
            ] {
                assert!(!is_authenticated_gfx942_wave64_scan_instance_v1(
                    tcx, instance, target
                ));
            }
        }
        for scalar in [
            tcx.types.u64,
            tcx.types.i64,
            tcx.types.f64,
            tcx.types.bool,
            tcx.types.unit,
        ] {
            let instance = Instance::new_raw(helper, tcx.mk_args(&[scalar.into()]));
            assert!(!is_authenticated_gfx942_wave64_scan_instance_v1(
                tcx, instance, target
            ));
        }
        for args in [
            rustc_middle::ty::List::empty(),
            tcx.mk_args(&[tcx.types.u32.into(), tcx.types.u32.into()]),
        ] {
            assert!(!is_authenticated_gfx942_wave64_scan_instance_v1(
                tcx,
                Instance::new_raw(helper, args),
                target,
            ));
        }
        let mut forged = 0;
        for local in tcx.hir_body_owners() {
            let def_id = local.to_def_id();
            if tcx.def_kind(def_id) != DefKind::Fn
                || tcx.item_name(def_id).as_str() != "forged_scan"
            {
                continue;
            }
            forged += 1;
            let signature = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
            assert_eq!(signature.safety, Safety::Unsafe);
            assert_eq!(signature.inputs().len(), 3);
            assert_eq!(signature.inputs()[0], tcx.types.u32);
            assert!(
                matches!(signature.inputs()[2].kind(), TyKind::Param(parameter) if parameter.index == 0)
            );
            assert_eq!(signature.output(), signature.inputs()[2]);
            assert!(validate_definition(tcx, def_id, &origin).is_err());
            assert_ne!(
                classify(tcx, def_id),
                Some(TrustedDeviceItem::Gfx942Wave64InclusiveScanHelper)
            );
            assert!(!is_authenticated_gfx942_wave64_scan_instance_v1(
                tcx,
                Instance::new_raw(def_id, tcx.mk_args(&[tcx.types.u32.into()])),
                target,
            ));
        }
        assert_eq!(
            forged, 1,
            "actual same-signature local impostor was not checked"
        );
    }
}
