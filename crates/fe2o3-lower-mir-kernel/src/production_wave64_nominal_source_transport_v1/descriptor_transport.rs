struct Wave64ReferenceCheckVisitorV1<'a> {
    budget: &'a mut dyn SemanticEmissionBudgetV1,
}
impl SemanticTransportVisitorV1 for Wave64ReferenceCheckVisitorV1<'_> {
    type Error = ProductionSemanticKirErrorV1;
    fn node(&mut self) -> Result<(), Self::Error> {
        self.budget.charge_work(1)
    }
    fn component(
        &mut self,
        _value: ValueId,
        ty: BorrowedTransportTypeV1<'_>,
    ) -> Result<(), Self::Error> {
        self.budget.charge_work(1)?;
        match ty {
            BorrowedTransportTypeV1::Existing(Type::Scalar(ScalarType::U32)) => Ok(()),
            _ => Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch),
        }
    }
    fn invalid(_: &'static str) -> Self::Error {
        ProductionSemanticKirErrorV1::CorrespondenceMismatch
    }
}
impl CheckedWave64ReferenceLocalsV1<'_> {
    fn arithmetic_lane_v1(
        &self,
        local: u32,
        binding: &SemanticValueBindingV1,
    ) -> Result<Option<ValueId>, ProductionSemanticKirErrorV1> {
        if self.rank_locals.contains(&local) {
            let SemanticValueBindingV1::Value { id, ty } = binding else {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            };
            if *ty != Type::Scalar(ScalarType::U32) {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            return Ok(Some(*id));
        }
        if let Some(Wave64NominalReferenceDescriptorV1 {
            lane_field: Some(field),
            ..
        }) = self.locals.get(&local)
        {
            let SemanticValueBindingV1::Aggregate(fields) = binding else {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            };
            let Some(SemanticValueBindingV1::Value { id, ty }) = fields.get(*field as usize) else {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            };
            if *ty != Type::Scalar(ScalarType::U32) {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            return Ok(Some(*id));
        }
        Ok(None)
    }
}

// A descriptor is coordinates only. Each actual consumer borrows a freshly
// rederived checked view on the SAME ledger, and cannot copy this guard out.
enum LiveWave64DescriptorSourceV1<'v, 'o> {
    Helper(&'v wave64_reference_call_v1::CheckedReferenceHelperV1<'o>),
    Borrow(&'v wave64_reference_call_v1::CheckedReferenceBorrowV1<'o>),
    ContextBorrow(&'v wave64_reference_call_v1::CheckedContextBorrowV1<'o>),
}
struct CheckedWave64DescriptorV1<'v, 'o> {
    descriptor: Wave64NominalReferenceDescriptorV1,
    source: LiveWave64DescriptorSourceV1<'v, 'o>,
}
impl CheckedWave64DescriptorV1<'_, '_> {
    fn transport_types_v1(&self) -> Vec<Type> {
        self.descriptor.transport_types_v1()
    }
    fn reconstruct_v1(
        &self,
        types: &[SemanticTypeDeclV1],
        ty: SemanticTypeIdV1,
        values: &[ValueDef],
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.descriptor.reconstruct_v1(types, ty, values)
    }
    fn visit_values_v1<V: SemanticTransportVisitorV1>(
        &self,
        binding: &SemanticValueBindingV1,
        expected: &[Type],
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.descriptor.visit_values_v1(binding, expected, visitor)
    }
    fn admits_dereference_v1(
        &self,
        function: SemanticFunctionIdV1,
        local: u32,
        ty: SemanticTypeIdV1,
        projection: &SemanticProjectionV1,
    ) -> bool {
        self.descriptor
            .admits_dereference_v1(function, local, ty, projection)
    }
}
impl CheckedWave64ReferenceLocalsV1<'_> {
    fn with_descriptor_v1(
        &self,
        descriptor: Wave64NominalReferenceDescriptorV1,
        budget: &mut dyn SemanticEmissionBudgetV1,
        inspect: &mut dyn FnMut(
            &CheckedWave64DescriptorV1<'_, '_>,
            &mut dyn SemanticEmissionBudgetV1,
        ) -> Result<(), ProductionSemanticKirErrorV1>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let owner = self
            .owner
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        if descriptor.function != self.function
            || self.locals.get(&descriptor.local) != Some(&descriptor)
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        let borrow = self
            .borrows
            .iter()
            .find_map(|(site, candidate)| (*candidate == descriptor).then_some(*site));
        let checked = if let Some((block, statement)) = borrow {
            budget.checked_wave64_borrow_v1(
                owner,
                self.function,
                SemanticBlockIdV1::from_index(block),
                statement,
                &mut |view, budget| {
                    if !std::ptr::eq(view.owner(), owner)
                        || Wave64NominalReferenceDescriptorV1::from_checked_borrow_v1(view)?
                            != descriptor
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    require_closed_wave64_reference_holder_v1(
                        owner,
                        self.function,
                        descriptor.local,
                        Some((view.block(), view.statement())),
                        budget,
                    )?;
                    inspect(
                        &CheckedWave64DescriptorV1 {
                            descriptor,
                            source: LiveWave64DescriptorSourceV1::Borrow(view),
                        },
                        budget,
                    )
                },
            )?
        } else if let Some((block, statement)) = self
            .context_borrows
            .iter()
            .find_map(|(site, candidate)| (*candidate == descriptor).then_some(*site))
        {
            budget.checked_wave64_context_borrow_v1(
                owner,
                self.function,
                SemanticBlockIdV1::from_index(block),
                statement,
                &mut |view, budget| {
                    if !std::ptr::eq(view.owner(), owner)
                        || Wave64NominalReferenceDescriptorV1::from_checked_context_borrow_v1(view)?
                            != descriptor
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    require_closed_wave64_reference_holder_v1(
                        owner,
                        self.function,
                        descriptor.local,
                        Some((view.block(), view.statement())),
                        budget,
                    )?;
                    inspect(
                        &CheckedWave64DescriptorV1 {
                            descriptor,
                            source: LiveWave64DescriptorSourceV1::ContextBorrow(view),
                        },
                        budget,
                    )
                },
            )?
        } else {
            budget.checked_wave64_helper_v1(owner, self.function, &mut |view, budget| {
                if !std::ptr::eq(view.representative().owner(), owner) {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let mut matched = false;
                for argument in 0..2 {
                    if Wave64NominalReferenceDescriptorV1::from_checked_formal_v1(view, argument)?
                        == Some(descriptor)
                    {
                        if matched {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        matched = true;
                    }
                }
                if !matched {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                require_closed_wave64_reference_holder_v1(
                    owner,
                    self.function,
                    descriptor.local,
                    None,
                    budget,
                )?;
                inspect(
                    &CheckedWave64DescriptorV1 {
                        descriptor,
                        source: LiveWave64DescriptorSourceV1::Helper(view),
                    },
                    budget,
                )
            })?
        };
        if !checked {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        Ok(())
    }
    fn remove_checked_retained_snapshots_v1(
        &self,
        candidates: &mut BTreeSet<u32>,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let Some(owner) = self.owner else {
            return Ok(());
        };
        for &local in &self.retained_snapshots {
            let mut validated = false;
            for &(block, statement) in self.borrows.keys() {
                budget.checked_wave64_borrow_v1(
                    owner,
                    self.function,
                    SemanticBlockIdV1::from_index(block),
                    statement,
                    &mut |view, budget| {
                        if !std::ptr::eq(view.owner(), owner) {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        let SemanticRvalueKindV1::Borrow { place, .. } =
                            view.source_rvalue().kind()
                        else {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        };
                        if place.local().index() == local {
                            let descriptor =
                                Wave64NominalReferenceDescriptorV1::from_checked_borrow_v1(view)?;
                            require_closed_wave64_reference_holder_v1(
                                owner,
                                self.function,
                                descriptor.local,
                                Some((view.block(), view.statement())),
                                budget,
                            )?;
                            candidates.remove(&local);
                            validated = true;
                        }
                        Ok(())
                    },
                )?;
            }
            for &(block, statement) in self.context_borrows.keys() {
                budget.checked_wave64_context_borrow_v1(
                    owner,
                    self.function,
                    SemanticBlockIdV1::from_index(block),
                    statement,
                    &mut |view, budget| {
                        let (function, _, issuer_local) = view.issuer();
                        if !std::ptr::eq(view.owner(), owner) || function != self.function {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        if issuer_local.index() == local {
                            let descriptor =
                                Wave64NominalReferenceDescriptorV1::from_checked_context_borrow_v1(
                                    view,
                                )?;
                            require_closed_wave64_reference_holder_v1(
                                owner,
                                self.function,
                                descriptor.local,
                                Some((view.block(), view.statement())),
                                budget,
                            )?;
                            candidates.remove(&local);
                            validated = true;
                        }
                        Ok(())
                    },
                )?;
            }
            if !validated {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        }
        Ok(())
    }
    fn seed_arithmetic_bound_v1(
        &self,
        local: u32,
        binding: &SemanticValueBindingV1,
        budget: &mut dyn SemanticEmissionBudgetV1,
        bounds: &mut BTreeMap<ValueId, u128>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if self.rank_locals.contains(&local) {
            let owner = self
                .owner
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            let checked =
                budget.checked_wave64_helper_v1(owner, self.function, &mut |view, budget| {
                    if !std::ptr::eq(view.representative().owner(), owner)
                        || !matches!(
                            view.representative().argument(0),
                            Some(wave64_reference_call_v1::ArgumentV1::Rank { .. })
                        )
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    if owner.source_semantic().functions()[self.function.index() as usize]
                        .locals()
                        .get(local as usize)
                        .is_none_or(|row| row.role() != SemanticLocalRoleV1::Argument(0))
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    require_unwritten_wave64_local_v1(owner, self.function, local, None, budget)?;
                    let Some(value) = self.arithmetic_lane_v1(local, binding)? else {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    };
                    bounds.insert(value, 64);
                    Ok(())
                })?;
            if !checked {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        } else if let Some(descriptor) = self.locals.get(&local).copied() {
            self.with_descriptor_v1(descriptor, budget, &mut |guard, _| {
                if guard.descriptor.lane_field.is_some() {
                    let Some(value) = self.arithmetic_lane_v1(local, binding)? else {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    };
                    bounds.insert(value, 64);
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}
impl SemanticFunctionLoweringV1<'_, '_> {
    fn checked_wave64_deref_v1(
        &mut self,
        local: u32,
        ty: SemanticTypeIdV1,
        projection: &SemanticProjectionV1,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        let Some(descriptor) = self.wave64_references.locals.get(&local).copied() else {
            return Ok(false);
        };
        let budget = self
            .emission_work
            .as_deref_mut()
            .ok_or(ArgumentResourceV1::Accounting)?;
        let mut admitted = false;
        self.wave64_references
            .with_descriptor_v1(descriptor, budget, &mut |guard, _| {
                admitted =
                    guard.admits_dereference_v1(self.semantic_function, local, ty, projection);
                Ok(())
            })?;
        Ok(admitted)
    }
}
