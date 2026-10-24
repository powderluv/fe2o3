/// Plain scalar payloads carry their semantic type, not tensor issuance authority.
/// The physical value remains in semantic MIR and is transported by the existing
/// typed KIR pipeline load/store lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProjectedPipelinePayloadV1 {
    Scalar(SemanticTypeIdV1),
    Mfma {
        element: SemanticTypeIdV1,
        operand: ProjectedMfmaOperandV1,
    },
}

fn plain_workgroup_pipeline_scalar_v1(
    types: &[SemanticTypeDeclV1],
    element: SemanticTypeIdV1,
) -> bool {
    let Some(declaration) = types.get(element.index() as usize) else {
        return false;
    };
    if !matches!(
        declaration.rust_type_kind(),
        SemanticRustTypeKindV1::Ordinary
            | SemanticRustTypeKindV1::Usize
            | SemanticRustTypeKindV1::Isize
    ) {
        return false;
    }
    let bits = match declaration.shape() {
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
            bits: bits @ (8 | 16 | 32 | 64),
            ..
        })
        | SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float {
            bits: bits @ (32 | 64),
        }) => *bits,
        _ => return false,
    };
    declaration
        .layout()
        .size_bytes()
        .and_then(|bytes| bytes.checked_mul(8))
        == Some(u64::from(bits))
}

fn collect_workgroup_pipeline_payloads_v1(
    callables: &[SemanticCallableDeclV1],
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
    pipeline_owners: &[Option<usize>],
    entries: &[Option<ProjectedCapabilityStateV1>],
    work: &mut usize,
) -> Result<HashMap<usize, ProjectedMfmaOperandV1>, ProductionRankedProjectionErrorV1> {
    let mut payloads = HashMap::new();
    let mut elements = HashMap::new();
    for (block_index, block) in function.blocks().iter().enumerate() {
        let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
            continue;
        };
        let Some(SemanticCallableDeclV1::CompilerIntrinsic { operation, .. }) =
            callables.get(call.callee().index() as usize)
        else {
            continue;
        };
        let (pipeline, element, write) = match operation {
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineWrite { pipeline, element } => {
                (*pipeline, *element, true)
            }
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineRead { pipeline, element } => {
                (*pipeline, *element, false)
            }
            _ => continue,
        };
        let Some(entry) = entries.get(block_index).and_then(Option::as_ref) else {
            continue;
        };
        charge_capability_dataflow_work_v1(work, entry.len().saturating_add(1))?;
        let mut state = try_clone_capability_state_v1(entry)?;
        transfer_capability_statements_v1(
            function,
            block_index,
            &mut state,
            enum_payload_dominance,
        )?;
        let (receiver, value) = match (write, call.arguments()) {
            (true, [receiver, _, _, value]) => (receiver, Some(value)),
            (false, [receiver, _, _])
                if call
                    .destination()
                    .is_some_and(|destination| destination.place().ty() == element) =>
            {
                (receiver, None)
            }
            _ => {
                return Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "a workgroup pipeline access lacks its exact receiver, epoch, index, and payload type",
                ));
            }
        };
        if value.is_some_and(|value| value.ty() != element) {
            return Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a workgroup pipeline write payload changed semantic type",
            ));
        }
        let owner = raw_operand_place(receiver)
            .and_then(|place| pipeline_owners.get(place.local().index() as usize))
            .copied()
            .flatten()
            .ok_or(ProductionRankedProjectionErrorV1::Incomplete(
                "a workgroup pipeline access lacks one compiler-owned origin",
            ))?;
        if function
            .locals()
            .get(owner)
            .is_none_or(|local| local.ty() != pipeline)
        {
            return Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a workgroup pipeline access changed its compiler-owned pipeline type",
            ));
        }
        if elements
            .insert(owner, element)
            .is_some_and(|existing| existing != element)
        {
            return Err(ProductionRankedProjectionErrorV1::Incomplete(
                "one workgroup pipeline is used with inconsistent payload types",
            ));
        }
        let Some(value) = value else {
            continue;
        };
        let payload = if plain_workgroup_pipeline_scalar_v1(types, element) {
            if capability_known_origin_v1(&state, value).is_some() {
                return Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "a scalar workgroup pipeline payload carries an incompatible capability origin",
                ));
            }
            ProjectedPipelinePayloadV1::Scalar(element)
        } else {
            let Some(ProjectedCapabilityOriginV1::Operand(operand)) =
                capability_known_origin_v1(&state, value)
            else {
                return Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "a workgroup pipeline write lacks one authenticated typed payload",
                ));
            };
            ProjectedPipelinePayloadV1::Mfma { element, operand }
        };
        match payloads.insert(owner, payload) {
            None => {}
            Some(existing) if existing == payload => {}
            Some(_) => {
                return Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "one workgroup pipeline receives incompatible typed payload layouts",
                ));
            }
        }
    }
    charge_capability_dataflow_work_v1(work, payloads.len())?;
    // Only fragments carry tensor provenance into subsequent capability reads.
    // Scalar reads remain ordinary typed values; never mint an MFMA origin for them.
    Ok(payloads
        .into_iter()
        .filter_map(|(owner, payload)| match payload {
            ProjectedPipelinePayloadV1::Mfma { operand, .. } => Some((owner, operand)),
            ProjectedPipelinePayloadV1::Scalar(_) => None,
        })
        .collect())
}
