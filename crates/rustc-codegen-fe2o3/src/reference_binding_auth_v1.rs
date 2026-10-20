//! Live rustc reference authentication and descriptive signature projection.

use super::*;

pub(crate) fn authenticate_reference_binding_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    registration_path: String,
    logical_kernel_name: String,
    kernel: Instance<'tcx>,
    reference: Instance<'tcx>,
    work: &mut SourceClosureWorkV1,
) -> Result<AuthenticatedReferenceEffectBindingV1, ReferenceBindingErrorV1> {
    authenticate_reference_binding_with_origin_v1(
        tcx,
        ReferenceBindingOriginV1::SourceRegistration(registration_path),
        logical_kernel_name,
        kernel,
        reference,
        work,
    )
}

pub(crate) fn authenticate_reference_binding_with_origin_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    origin: ReferenceBindingOriginV1,
    logical_kernel_name: String,
    kernel: Instance<'tcx>,
    reference: Instance<'tcx>,
    work: &mut SourceClosureWorkV1,
) -> Result<AuthenticatedReferenceEffectBindingV1, ReferenceBindingErrorV1> {
    let meter = &ReferenceExtractionWorkV1::borrowed(work);
    authenticate_safe_local_reference_v1(meter, tcx, reference)?;
    let (signature_preimage, relations) = logical_abi_relation_v1(meter, tcx, kernel, reference)?;
    let effect_ir = lower_reference_effect_ir_v1(meter, tcx, reference, relations)?;
    meter.ir_hash(&effect_ir)?;
    let effect_ir_sha256 = effect_ir.canonical_sha256_v1();
    meter.charge(meter.effects(&effect_ir.observable_output_effects)?)?;
    let observable_output_writes = effect_ir.observable_output_effects.clone();
    meter.charge(effect_ir.relations.len())?;
    if effect_ir.relations.iter().any(|relation| {
        matches!(
            relation,
            ReferenceArgumentRelationV1::DisjointOutputSlice { .. }
                | ReferenceArgumentRelationV1::DisjointOutputCoordinate { .. }
        )
    }) && observable_output_writes.is_empty()
    {
        return Err(ReferenceBindingErrorV1::new(
            "safe Rust reference has a logical output but reference-effect V1 found no observable output write",
        ));
    }
    meter.rows::<AuthenticatedReferenceEffectBindingV1>(1)?;
    Ok(AuthenticatedReferenceEffectBindingV1 {
        origin,
        logical_kernel_name,
        kernel: function_identity_v1(meter, tcx, kernel)?,
        reference: function_identity_v1(meter, tcx, reference)?,
        signature_preimage,
        effect_ir_sha256,
        effect_ir,
        observable_output_writes,
    })
}

pub(super) fn function_identity_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Result<ReferenceFunctionIdentityV1, ReferenceBindingErrorV1> {
    charge_reference_source_v1(tcx, instance, meter)?;
    let identities = canonical_function_identities_v1(tcx, instance);
    Ok(ReferenceFunctionIdentityV1 {
        def_path_hash: tcx.def_path_hash(instance.def_id()).0.to_le_bytes(),
        function_sha256: *identities.function().as_bytes(),
        item_definition_sha256: *identities.item_definition().as_bytes(),
        monomorphization_sha256: *identities.monomorphization().as_bytes(),
        generic_type_arguments_sha256: *identities.generic_type_arguments().as_bytes(),
        const_generic_arguments_sha256: *identities.const_generic_arguments().as_bytes(),
        rustc_mir_body_sha256: rustc_mir_body_sha256_v1(tcx, instance),
    })
}

pub(crate) fn instantiated_signature<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> rustc_middle::ty::FnSig<'tcx> {
    tcx.instantiate_bound_regions_with_erased(
        tcx.fn_sig(instance.def_id())
            .instantiate(tcx, instance.args),
    )
}

fn logical_abi_relation_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    kernel: Instance<'tcx>,
    reference: Instance<'tcx>,
) -> Result<
    (
        ReferenceLogicalSignaturePreimageV1,
        Vec<ReferenceArgumentRelationV1>,
    ),
    ReferenceBindingErrorV1,
> {
    let kernel_signature = instantiated_signature(tcx, kernel);
    let reference_signature = instantiated_signature(tcx, reference);
    let preimage = extract_reference_signature_preimage_metered_v1(
        meter,
        tcx,
        kernel_signature,
        reference_signature,
    )?;
    meter.charge(preimage.reference_inputs().len())?;
    let derived = preimage.derive_relations_v1().map_err(|error| {
        reference_signature_error_v1(error, kernel_signature, reference_signature)
    })?;
    let mut relations = reserve_reference_signature_rows_v1(meter, derived.len())?;
    for raw in 0..derived.len() {
        let raw = u32::try_from(raw)
            .map_err(|_| ReferenceBindingErrorV1::new("reference argument index exceeds u32"))?;
        meter.charge(1)?;
        relations.push(derived.relation_at_raw_argument_v1(raw).ok_or_else(|| {
            ReferenceBindingErrorV1::new("checked reference signature relation lost an argument")
        })?);
    }
    Ok((preimage, relations))
}

/// Descriptive signature fields, not source/provider authority.
fn extract_reference_signature_preimage_metered_v1<'tcx>(
    meter: &ReferenceExtractionWorkV1<'_>,
    tcx: TyCtxt<'tcx>,
    kernel: rustc_middle::ty::FnSig<'tcx>,
    reference: rustc_middle::ty::FnSig<'tcx>,
) -> Result<ReferenceLogicalSignaturePreimageV1, ReferenceBindingErrorV1> {
    meter.charge(1)?;
    use fe2o3_mir_model::semantic_mir_v1::{SemanticExternAbiV1, SemanticFunctionSafetyV1};

    if reference.abi != ExternAbi::Rust {
        return Err(ReferenceBindingErrorV1::new(
            ReferenceSignatureErrorV1::InvalidReferenceAbi.to_string(),
        ));
    }
    let result = if reference.output() == tcx.types.unit {
        ReferenceReturnShapeV1::Unit
    } else {
        ReferenceReturnShapeV1::NonUnit
    };
    let safety = match reference.safety {
        Safety::Safe => SemanticFunctionSafetyV1::Safe,
        Safety::Unsafe => SemanticFunctionSafetyV1::Unsafe,
    };
    let axes = ReferenceLogicalSignaturePreimageV1::check_header_v1(
        kernel.inputs().len(),
        reference.inputs().len(),
        result,
        SemanticExternAbiV1::Rust,
        safety,
        reference.c_variadic,
    )
    .map_err(|error| reference_signature_error_v1(error, kernel, reference))?;

    let mut kernel_inputs = reserve_reference_signature_rows_v1(meter, kernel.inputs().len())?;
    for (argument, ty) in kernel.inputs().iter().copied().enumerate() {
        meter.charge(1)?;
        kernel_inputs.push(
            extract_reference_signature_input_v1(tcx, ty).map_err(|reason| {
                ReferenceBindingErrorV1::new(format!(
                    "kernel argument {} type '{ty}' has no reference ABI relation: {reason}",
                    argument + 1,
                ))
            })?,
        );
    }
    let mut reference_inputs =
        reserve_reference_signature_rows_v1(meter, reference.inputs().len())?;
    for (argument, ty) in reference.inputs().iter().copied().enumerate() {
        meter.charge(1)?;
        reference_inputs.push(extract_reference_signature_input_v1(tcx, ty).map_err(|reason| {
            if argument < axes {
                ReferenceBindingErrorV1::new(format!(
                    "safe Rust point-reference coordinate argument {} must be usize, found '{ty}': {reason}",
                    argument + 1,
                ))
            } else {
                ReferenceBindingErrorV1::new(format!(
                    "safe Rust reference logical ABI mismatch at argument {}: kernel '{}', reference '{ty}': {reason}",
                    argument - axes + 1,
                    kernel.inputs()[argument - axes],
                ))
            }
        })?);
    }
    ReferenceLogicalSignaturePreimageV1::new(
        kernel_inputs.into_boxed_slice(),
        reference_inputs.into_boxed_slice(),
        result,
        SemanticExternAbiV1::Rust,
        safety,
        reference.c_variadic,
    )
    .map_err(|error| reference_signature_error_v1(error, kernel, reference))
}

fn reserve_reference_signature_rows_v1<T>(
    meter: &ReferenceExtractionWorkV1<'_>,
    count: usize,
) -> Result<Vec<T>, ReferenceBindingErrorV1> {
    meter.rows::<T>(count)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(count).map_err(|_| {
        ReferenceBindingErrorV1::new("reference logical signature row allocation failed")
    })?;
    if rows.capacity() != count {
        return Err(ReferenceBindingErrorV1::new(
            "reference logical signature row allocation exceeded requested capacity",
        ));
    }
    Ok(rows)
}

fn reference_signature_error_v1<'tcx>(
    error: ReferenceSignatureErrorV1,
    kernel: rustc_middle::ty::FnSig<'tcx>,
    reference: rustc_middle::ty::FnSig<'tcx>,
) -> ReferenceBindingErrorV1 {
    match error {
        ReferenceSignatureErrorV1::InvalidPointCoordinate { reference_argument } => {
            if let Some(ty) = reference.inputs().get(reference_argument) {
                return ReferenceBindingErrorV1::new(format!(
                    "safe Rust point-reference coordinate argument {} must be usize, found '{ty}'",
                    reference_argument + 1,
                ));
            }
        }
        ReferenceSignatureErrorV1::ArgumentMismatch {
            kernel_argument,
            reference_argument,
        } => {
            if let (Some(kernel_ty), Some(reference_ty)) = (
                kernel.inputs().get(kernel_argument),
                reference.inputs().get(reference_argument),
            ) {
                return logical_abi_mismatch(kernel_argument, *kernel_ty, *reference_ty);
            }
        }
        ReferenceSignatureErrorV1::UnsupportedKernelArgument { argument } => {
            if let Some(ty) = kernel.inputs().get(argument) {
                return ReferenceBindingErrorV1::new(format!(
                    "kernel argument {} type '{ty}' has no reference ABI relation",
                    argument + 1,
                ));
            }
        }
        _ => {}
    }
    ReferenceBindingErrorV1::new(error.to_string())
}

fn logical_abi_mismatch(
    index: usize,
    kernel_ty: Ty<'_>,
    reference_ty: Ty<'_>,
) -> ReferenceBindingErrorV1 {
    ReferenceBindingErrorV1::new(format!(
        "safe Rust reference logical ABI mismatch at argument {}: kernel '{kernel_ty}', reference '{reference_ty}'",
        index + 1,
    ))
}

pub(super) fn scalar_type_v1(ty: Ty<'_>) -> Option<ReferenceScalarTypeV1> {
    use rustc_middle::ty::{FloatTy, IntTy, UintTy};
    Some(match *ty.kind() {
        TyKind::Bool => ReferenceScalarTypeV1::Bool,
        TyKind::Uint(UintTy::U8) => ReferenceScalarTypeV1::U8,
        TyKind::Uint(UintTy::U16) => ReferenceScalarTypeV1::U16,
        TyKind::Uint(UintTy::U32) => ReferenceScalarTypeV1::U32,
        TyKind::Uint(UintTy::U64) => ReferenceScalarTypeV1::U64,
        TyKind::Uint(UintTy::Usize) => ReferenceScalarTypeV1::Usize,
        TyKind::Int(IntTy::I8) => ReferenceScalarTypeV1::I8,
        TyKind::Int(IntTy::I16) => ReferenceScalarTypeV1::I16,
        TyKind::Int(IntTy::I32) => ReferenceScalarTypeV1::I32,
        TyKind::Int(IntTy::I64) => ReferenceScalarTypeV1::I64,
        TyKind::Int(IntTy::Isize) => ReferenceScalarTypeV1::Isize,
        TyKind::Float(FloatTy::F32) => ReferenceScalarTypeV1::F32,
        TyKind::Float(FloatTy::F64) => ReferenceScalarTypeV1::F64,
        _ => return None,
    })
}

fn extract_reference_signature_input_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<ReferenceSignatureInputV1, &'static str> {
    use fe2o3_mir_model::semantic_mir_v1::SemanticMutabilityV1;

    if let Some(scalar) = scalar_type_v1(ty) {
        return Ok(ReferenceSignatureInputV1::Scalar(scalar));
    }
    if let TyKind::Ref(region, pointee, mutability) = *ty.kind() {
        let pointee = if let TyKind::Slice(element) = *pointee.kind() {
            ReferencePointeeV1::Slice(
                scalar_type_v1(element).ok_or("unsupported reference slice element")?,
            )
        } else {
            ReferencePointeeV1::Scalar(
                scalar_type_v1(pointee).ok_or("unsupported reference pointee")?,
            )
        };
        let region = match region.kind() {
            rustc_middle::ty::ReErased => ReferenceRegionV1::Erased,
            rustc_middle::ty::ReStatic => ReferenceRegionV1::Static,
            _ => {
                return Err("reference region cannot be represented without losing exact equality");
            }
        };
        return Ok(ReferenceSignatureInputV1::Reference {
            region,
            mutability: match mutability {
                Mutability::Not => SemanticMutabilityV1::Immutable,
                Mutability::Mut => SemanticMutabilityV1::Mutable,
            },
            pointee,
        });
    }
    let TyKind::Adt(definition, arguments) = *ty.kind() else {
        return Err("unsupported logical signature input type");
    };
    let carrier = match trusted_device_items::classify(tcx, definition.did()) {
        Some(TrustedDeviceItem::DisjointSlice) => ReferenceCarrierV1::DisjointSlice,
        Some(TrustedDeviceItem::WriteOnlyDisjointSlice) => {
            ReferenceCarrierV1::WriteOnlyDisjointSlice
        }
        _ => return Err("nominal type is not an authenticated reference output carrier"),
    };
    let element = arguments
        .first()
        .and_then(|argument| argument.as_type())
        .and_then(scalar_type_v1)
        .ok_or("unsupported nominal output element")?;
    Ok(ReferenceSignatureInputV1::NominalOutput { carrier, element })
}
