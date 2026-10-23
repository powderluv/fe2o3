// Arithmetic-only LOCAL descriptor. This is not compiler-issued type/EXEC
// authority: its owner must remain attached to the exact function, and each
// insertion is made from a borrowed checked helper or original BorrowSite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Wave64NominalReferenceDescriptorV1 {
    function: SemanticFunctionIdV1,
    local: u32,
    reference_type: SemanticTypeIdV1,
    pointee: SemanticTypeIdV1,
    lane_field: Option<u32>,
}
impl Wave64NominalReferenceDescriptorV1 {
    fn from_checked_formal_v1(
        source: &wave64_reference_call_v1::CheckedReferenceHelperV1<'_>,
        argument: usize,
    ) -> Result<Option<Self>, ProductionSemanticKirErrorV1> {
        let relation = source.representative();
        let declaration = relation.helper_declaration();
        let Some(role) = relation.argument(argument) else {
            return Ok(None);
        };
        let (reference_type, pointee, lane_field) = match role {
            wave64_reference_call_v1::ArgumentV1::Snapshot {
                reference_type,
                snapshot_type,
                lane_field,
            } => (reference_type, snapshot_type, Some(lane_field)),
            wave64_reference_call_v1::ArgumentV1::Context {
                reference_type,
                context_type,
            } => (reference_type, context_type, None),
            wave64_reference_call_v1::ArgumentV1::Rank { .. } => return Ok(None),
        };
        let mut local = None;
        for (index, row) in declaration.locals().iter().enumerate() {
            if row.role() == SemanticLocalRoleV1::Argument(argument as u32) {
                if row.ty() != reference_type || local.replace(index).is_some() {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
            }
        }
        let local =
            u32::try_from(local.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?)
                .map_err(|_| ArgumentResourceV1::Arithmetic)?;
        Ok(Some(Self {
            function: relation.helper(),
            local,
            reference_type,
            pointee,
            lane_field,
        }))
    }

    fn from_checked_borrow_v1(
        source: &wave64_reference_call_v1::CheckedReferenceBorrowV1<'_>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let function = source
            .owner()
            .source_semantic()
            .functions()
            .get(source.function().index() as usize)
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        let statement = function
            .blocks()
            .get(source.block().index() as usize)
            .and_then(|block| block.statements().get(source.statement() as usize))
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        };
        if !std::ptr::eq(assignment.value(), source.source_rvalue())
            || !assignment.destination().projections().is_empty()
            || assignment.destination().ty() != source.reference_type()
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        Ok(Self {
            function: source.function(),
            local: assignment.destination().local().index(),
            reference_type: source.reference_type(),
            pointee: source.snapshot_type(),
            lane_field: Some(source.lane_field()),
        })
    }

    fn from_checked_context_borrow_v1(
        source: &wave64_reference_call_v1::CheckedContextBorrowV1<'_>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let function =
            &source.owner().source_semantic().functions()[source.function().index() as usize];
        let Some(statement) = function
            .blocks()
            .get(source.block().index() as usize)
            .and_then(|block| block.statements().get(source.statement() as usize))
        else {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        };
        let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        };
        if !std::ptr::eq(assignment.value(), source.source_rvalue())
            || !assignment.destination().projections().is_empty()
            || assignment.destination().ty() != source.reference_type()
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        Ok(Self {
            function: source.function(),
            local: assignment.destination().local().index(),
            reference_type: source.reference_type(),
            pointee: source.context_type(),
            lane_field: None,
        })
    }
    fn exact_local_v1(
        self,
        function: SemanticFunctionIdV1,
        local: u32,
        ty: SemanticTypeIdV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if (self.function, self.local, self.reference_type) != (function, local, ty) {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        Ok(())
    }
    fn transport_types_v1(self) -> Vec<Type> {
        match self.lane_field {
            Some(_) => vec![Type::Scalar(ScalarType::U32)],
            None => Vec::new(),
        }
    }
    fn reconstruct_v1(
        self,
        types: &[SemanticTypeDeclV1],
        reference_type: SemanticTypeIdV1,
        values: &[ValueDef],
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        if reference_type != self.reference_type {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        let Some(SemanticTypeShapeV1::Pointer(pointer)) = types
            .get(reference_type.index() as usize)
            .map(SemanticTypeDeclV1::shape)
        else {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        };
        if pointer.kind() != SemanticPointerKindV1::Reference
            || pointer.mutability() != SemanticMutabilityV1::Immutable
            || pointer.metadata() != SemanticPointerMetadataV1::None
            || pointer.address_space() != 0
            || pointer.pointer_width_bits() != 64
            || pointer.pointee() != self.pointee
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        match self.lane_field {
            Some(lane_field) => {
                if values.len() != 1 || values[0].ty != Type::Scalar(ScalarType::U32) {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let result = binding_from_value_defs(types, self.pointee, values)?;
                let SemanticValueBindingV1::Aggregate(fields) = &result else {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                };
                if !matches!(fields.get(lane_field as usize), Some(SemanticValueBindingV1::Value { id, ty })
                    if *id == values[0].id && *ty == Type::Scalar(ScalarType::U32))
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                Ok(result)
            }
            None if values.is_empty()
                && types
                    .get(self.pointee.index() as usize)
                    .is_some_and(|ty| ty.layout().size_bytes() == Some(0)) =>
            {
                Ok(SemanticValueBindingV1::CollectiveContext)
            }
            None => Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch),
        }
    }
    fn visit_values_v1<V: SemanticTransportVisitorV1>(
        self,
        binding: &SemanticValueBindingV1,
        expected: &[Type],
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        match (self.lane_field, binding, expected) {
            (None, SemanticValueBindingV1::CollectiveContext, []) => Ok(()),
            (Some(field), SemanticValueBindingV1::Aggregate(fields), [expected]) => {
                let Some(SemanticValueBindingV1::Value { id, ty }) = fields.get(field as usize)
                else {
                    return Err(V::invalid(
                        "checked snapshot arithmetic lane binding differs",
                    ));
                };
                if *ty != Type::Scalar(ScalarType::U32) || !visitor.equal_types(ty, expected)? {
                    return Err(V::invalid(
                        "checked snapshot arithmetic lane carrier differs",
                    ));
                }
                // Every other source field was proved ZST by the checked source
                // relation; ensure no runtime value has appeared in transport.
                for (index, field_binding) in fields.iter().enumerate() {
                    visitor.node()?;
                    if index != field as usize {
                        visit_wave64_empty_source_zst_v1(field_binding, visitor, 0)?;
                    }
                }
                visitor.component(*id, BorrowedTransportTypeV1::Existing(ty))
            }
            _ => Err(V::invalid("checked Wave64 reference source role differs")),
        }
    }
    fn admits_dereference_v1(
        self,
        function: SemanticFunctionIdV1,
        local: u32,
        reference_type: SemanticTypeIdV1,
        projection: &SemanticProjectionV1,
    ) -> bool {
        (self.function, self.local, self.reference_type) == (function, local, reference_type)
            && projection.kind() == SemanticProjectionKindV1::Dereference
            && projection.result_type() == self.pointee
    }
}

fn visit_wave64_empty_source_zst_v1<V: SemanticTransportVisitorV1>(
    binding: &SemanticValueBindingV1,
    visitor: &mut V,
    depth: usize,
) -> Result<(), V::Error> {
    visitor.node()?;
    if depth >= 64 {
        return Err(V::invalid("checked snapshot ZST transport depth"));
    }
    match binding {
        SemanticValueBindingV1::Unit => Ok(()),
        SemanticValueBindingV1::Aggregate(fields) => {
            for field in fields {
                visit_wave64_empty_source_zst_v1(field, visitor, depth + 1)?;
            }
            Ok(())
        }
        _ => Err(V::invalid(
            "checked snapshot gained runtime state or a foreign role",
        )),
    }
}
