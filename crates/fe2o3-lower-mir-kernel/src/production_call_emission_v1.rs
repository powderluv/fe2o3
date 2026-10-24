include!("production_source_reference_call_transport_v26.rs");

struct DefinedCallArgumentSignatureV1<'a, 'scope> {
    projection: DefinedCallProjectionV29<'scope>,
    semantic_types: &'a [SemanticTypeIdV1],
    projections: &'a [HelperCallArgumentV1],
    parameter_types: Vec<Type>,
}

struct PreparedDefinedCallArgumentsV1<'scope> {
    source_bindings: Vec<SemanticValueBindingV1>,
    arguments: Vec<ValueId>,
    execution: Option<PreparedExecutionCallOriginV29<'scope>>,
}

impl<'a, 'service> SemanticFunctionLoweringV1<'a, 'service> {
    fn lower_return(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
    ) -> Result<Terminator, ProductionSemanticKirErrorV1> {
        let (values, components) = self.prepare_return_values_v1(block, operations)?;
        self.observe_execution_return_v1(block, &values)?;
        self.finish_execution_lifecycle_return_v29(block, operations)?;
        self.record_call_return_v1(block, SemanticKirCallReturnKindV1::Return { components })?;
        Ok(Terminator::Return { values })
    }

    fn prepare_return_values_v1(
        &mut self,
        block: SemanticBlockIdV1,
        operations: &mut Vec<Operation>,
    ) -> Result<(Vec<ValueId>, CallComponentSpanV1), ProductionSemanticKirErrorV1> {
        let return_local = self
            .function
            .locals()
            .iter()
            .position(|local| local.role() == SemanticLocalRoleV1::Return)
            .ok_or_else(|| {
                unsupported(
                    self.semantic_function.index(),
                    Some(block.index()),
                    None,
                    "function return local is missing",
                )
            })?;
        if self
            .legacy_retained_slot_v29(SemanticLocalIdV1::from_index(return_local as u32))?
            .is_some()
            && !matches!(
                self.types[self.function.locals()[return_local].ty().index() as usize].shape(),
                SemanticTypeShapeV1::Scalar(_) | SemanticTypeShapeV1::ValidityScalar(_)
            )
        {
            return Err(unsupported(
                self.semantic_function.index(),
                Some(block.index()),
                None,
                "aggregate helper return requires a whole SSA local",
            ));
        }
        self.consume_execution_return_source_v1(block, return_local)?;
        let inputs = if let Some(inputs) = self.source_reference_return_inputs_v29(return_local)? {
            inputs
        } else if self.result_types.is_empty() {
            Vec::new()
        } else {
            self.locals
                .get(return_local)
                .and_then(Option::as_ref)
                .ok_or(ProductionSemanticKirErrorV1::MissingLocalDefinition {
                    function: self.semantic_function.index(),
                    block: block.index(),
                    statement: None,
                    local: return_local as u32,
                })?
                .values()
                .map_err(|detail| {
                    unsupported(
                        self.semantic_function.index(),
                        Some(block.index()),
                        None,
                        detail,
                    )
                })?
        };
        if inputs.len() != self.result_types.len() {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        let first = self.call_returns.components.rows.len();
        let mut returned = argument_vec_v1(inputs.len())?;
        for (slot, (input, actual)) in inputs.into_iter().enumerate() {
            let expected = self.result_types[slot].clone();
            let (value, conversion) =
                if actual == Type::INDEX && expected == Type::Scalar(ScalarType::U64) {
                    let ordinal = call_operation_ordinal_v1(operations, block)?;
                    let value = self
                        .emit(
                            operations,
                            expected.clone(),
                            OperationKind::Cast {
                                kind: CastKind::Bitcast,
                                value: input,
                                to: expected,
                            },
                        )?
                        .value()
                        .map_err(|detail| {
                            unsupported(
                                self.semantic_function.index(),
                                Some(block.index()),
                                None,
                                detail,
                            )
                        })?
                        .0;
                    (value, Some(ordinal))
                } else if actual == expected {
                    (input, None)
                } else if source_descriptor_widening_v29(&actual, &expected) {
                    self.check_descriptor_return_v29(
                        block,
                        return_local,
                        slot,
                        input,
                        &actual,
                        &expected,
                    )?;
                    let ordinal = call_operation_ordinal_v1(operations, block)?;
                    let value = self.emit_id(
                        operations,
                        expected.clone(),
                        OperationKind::Cast {
                            kind: CastKind::SliceToGeneric,
                            value: input,
                            to: expected,
                        },
                    )?;
                    (value, Some(ordinal))
                } else {
                    return Err(unsupported(
                        self.semantic_function.index(),
                        Some(block.index()),
                        None,
                        "helper return local component type changed",
                    ));
                };
            self.call_returns
                .components
                .push(CallResultComponentV1::Return { input, conversion })?;
            returned.push(value);
        }
        let components = self.call_returns.component_span(first)?;
        Ok((returned, components))
    }

    fn prepare_defined_call_arguments_v1<'scope>(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        callee: SemanticFunctionIdV1,
        signature: DefinedCallArgumentSignatureV1<'_, 'scope>,
        operations: &mut Vec<Operation>,
    ) -> Result<PreparedDefinedCallArgumentsV1<'scope>, ProductionSemanticKirErrorV1> {
        if call.arguments().len() != signature.semantic_types.len()
            || signature.projections.len() != signature.parameter_types.len()
        {
            return Err(unsupported(
                self.semantic_function.index(),
                Some(block.index()),
                None,
                "defined call argument or result arity changed",
            ));
        }
        let execution = self.prepare_execution_call_origin_v29(block, call, callee, &signature)?;
        // Moving each source operand once precedes outer-tuple expansion.
        let mut source_bindings = if execution.is_some() {
            emission_vec_v1(
                call.arguments().len(),
                self.emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?,
            )?
        } else {
            let mut bindings = Vec::new();
            bindings
                .try_reserve_exact(call.arguments().len())
                .map_err(|_| ProductionSemanticKirErrorV1::AllocationFailure {
                    resource: ProductionSemanticKirResourceV1::AnalysisStorage,
                })?;
            bindings
        };
        for (index, (argument, expected)) in call
            .arguments()
            .iter()
            .zip(signature.semantic_types)
            .enumerate()
        {
            if semantic_operand_type(argument) != *expected {
                return Err(unsupported(
                    self.semantic_function.index(),
                    Some(block.index()),
                    None,
                    "defined call source argument type changed",
                ));
            }
            source_bindings.push(self.lower_source_operand_v29(
                block,
                None,
                Some(ExecutionOperandV29::CallArgument(
                    u32::try_from(index).map_err(|_| execution_availability_error_v29())?,
                )),
                argument,
                operations,
            )?);
        }
        if let Some(origin) = execution.as_ref() {
            let references = self.execution.as_ref().and_then(|cursor| cursor.references);
            let budget = self
                .emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            for (index, (binding, ty)) in source_bindings
                .iter()
                .zip(signature.semantic_types)
                .enumerate()
            {
                if let Some(references) = references {
                    if source_reference_call_argument_shape_v29(
                        references,
                        origin,
                        u32::try_from(index).map_err(|_| ArgumentResourceV1::Arithmetic)?,
                        binding,
                        budget,
                    )? {
                        continue;
                    }
                }
                execution_call_argument_shape_with_representation_v29(
                    self.types,
                    *ty,
                    index as u32,
                    binding,
                    signature.projections,
                    &signature.parameter_types,
                    if references.is_some() {
                        ExecutionCfgRepresentationV29::OriginalSource
                    } else {
                        ExecutionCfgRepresentationV29::LegacyAbi
                    },
                    budget,
                )?;
            }
        }
        let mut arguments = if execution.is_some() {
            emission_vec_v1(
                signature.parameter_types.len(),
                self.emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?,
            )?
        } else {
            let mut arguments = Vec::new();
            arguments
                .try_reserve_exact(signature.parameter_types.len())
                .map_err(|_| ProductionSemanticKirErrorV1::AllocationFailure {
                    resource: ProductionSemanticKirResourceV1::AnalysisStorage,
                })?;
            arguments
        };
        let mut flattened = None;
        for (parameter, (projection, expected)) in signature
            .projections
            .iter()
            .zip(signature.parameter_types)
            .enumerate()
        {
            let source = source_bindings
                .get(projection.source_argument as usize)
                .ok_or_else(|| {
                    unsupported(
                        self.semantic_function.index(),
                        Some(block.index()),
                        None,
                        "defined call projection has no source argument",
                    )
                })?;
            let binding = match (source, projection.tuple_field) {
                (SemanticValueBindingV1::Aggregate(fields), Some(field)) => {
                    fields.get(field as usize).ok_or_else(|| {
                        unsupported(
                            self.semantic_function.index(),
                            Some(block.index()),
                            None,
                            "defined call tuple field is missing",
                        )
                    })?
                }
                (_, None) => source,
                (_, Some(_)) => {
                    return Err(unsupported(
                        self.semantic_function.index(),
                        Some(block.index()),
                        None,
                        "defined call RustCall argument is not a tuple binding",
                    ));
                }
            };
            let function = self.semantic_function.index();
            let failure = |detail| unsupported(function, Some(block.index()), None, detail);
            if execution.is_some()
                && projection.component.is_none()
                && execution_is_direct_owner_type_v29(&expected)
            {
                let SemanticValueBindingV1::Value { id, ty } = binding else {
                    return Err(execution_call_error_v29());
                };
                if ty == &expected {
                    arguments.push(*id);
                    continue;
                }
                if !source_descriptor_widening_v29(ty, &expected) {
                    return Err(execution_call_error_v29());
                }
            }
            let (value, actual) = match projection.component {
                None => binding.value().map_err(failure)?,
                Some(component) => {
                    let key = (projection.source_argument, projection.tuple_field);
                    if flattened
                        .as_ref()
                        .is_none_or(|(previous, _)| *previous != key)
                    {
                        let values = if execution.is_some() {
                            let budget = self
                                .emission_work
                                .as_deref_mut()
                                .ok_or(ArgumentResourceV1::Accounting)?;
                            let mut physical = Vec::new();
                            if let Some(references) =
                                self.execution.as_ref().and_then(|cursor| cursor.references)
                            {
                                source_reference_values_v29(
                                    references,
                                    binding,
                                    &mut physical,
                                    &mut 0,
                                    budget,
                                )?;
                            } else {
                                execution_cfg_values_v29(binding, &mut physical, &mut 0, budget)?;
                            }
                            let mut values = emission_vec_v1(physical.len(), budget)?;
                            budget.charge_work(physical.len())?;
                            values.extend(physical.into_iter().map(|value| (value.id, value.ty)));
                            values
                        } else {
                            binding.values().map_err(failure)?
                        };
                        flattened = Some((key, values));
                    }
                    flattened
                        .as_ref()
                        .unwrap()
                        .1
                        .get(component)
                        .cloned()
                        .ok_or_else(|| failure("defined call aggregate component is missing"))?
                }
            };
            let value = if actual == Type::INDEX && expected == Type::Scalar(ScalarType::U64) {
                self.emit(
                    operations,
                    expected.clone(),
                    OperationKind::Cast {
                        kind: CastKind::Bitcast,
                        value,
                        to: expected,
                    },
                )?
                .value()
                .map_err(failure)?
                .0
            } else if actual == expected {
                value
            } else if matches!((&actual, &expected), (Type::Pointer(_), Type::Pointer(_))) {
                let origin = execution.as_ref().ok_or_else(execution_call_error_v29)?;
                self.check_reference_call_argument_v26(
                    block, call, origin, parameter, projection, binding, value, &actual, &expected,
                )?;
                self.emit_id(
                    operations,
                    expected.clone(),
                    OperationKind::Cast {
                        kind: CastKind::PointerToGeneric,
                        value,
                        to: expected,
                    },
                )?
            } else if source_descriptor_widening_v29(&actual, &expected) {
                let origin = execution.as_ref().ok_or_else(execution_call_error_v29)?;
                self.check_descriptor_call_argument_v29(
                    block, call, origin, parameter, projection, binding, value, &actual, &expected,
                )?;
                self.emit_id(
                    operations,
                    expected.clone(),
                    OperationKind::Cast {
                        kind: CastKind::SliceToGeneric,
                        value,
                        to: expected,
                    },
                )?
            } else {
                return Err(
                    ProductionSemanticKirErrorV1::DefinedCallArgumentTypeMismatch {
                        function: self.semantic_function.index(),
                        callee: callee.index(),
                        block: block.index(),
                        parameter,
                        source_argument: projection.source_argument,
                        tuple_field: projection.tuple_field,
                        component: projection.component,
                        expected,
                        actual,
                    },
                );
            };
            arguments.push(value);
        }
        Ok(PreparedDefinedCallArgumentsV1 {
            source_bindings,
            arguments,
            execution,
        })
    }

    fn lower_defined_call(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        callee: SemanticFunctionIdV1,
        operations: &mut Vec<Operation>,
    ) -> Result<Terminator, ProductionSemanticKirErrorV1> {
        let mut prefix = self.prepare_defined_call_prefix_v1(block, call, callee, operations)?;
        let parameters = std::mem::take(&mut prefix.signature.parameter_types);
        let (callee_id, arguments, nominal) = if self.execution_calls.is_some() {
            self.prepare_execution_defined_call_v29(
                block,
                call,
                callee,
                &prefix.signature.parameter_semantic_types,
                &prefix.signature.call_arguments,
                parameters,
                operations,
            )?
        } else {
            let callee_id = prefix
                .ordinary_target
                .take()
                .ok_or_else(execution_call_error_v29)?;
            let PreparedDefinedCallArgumentsV1 {
                arguments,
                source_bindings,
                ..
            } = self.prepare_defined_call_arguments_v1(
                block,
                call,
                callee,
                DefinedCallArgumentSignatureV1 {
                    projection: DefinedCallProjectionV29::Ordinary,
                    semantic_types: &prefix.signature.parameter_semantic_types,
                    projections: &prefix.signature.call_arguments,
                    parameter_types: parameters,
                },
                operations,
            )?;
            if prefix.signature.bf16_nominal {
                bf16_check_call_bindings_v1(&source_bindings)?;
            }
            if prefix.signature.wave64_nominal {
                self.check_checked_wave64_call_bindings_v1(block, call, callee, &source_bindings, &arguments)?;
            }
            (callee_id, arguments, Vec::new())
        };
        let (terminator, consumed_credit) = self.finish_defined_call_prefix_v1(
            call, prefix, callee_id, arguments, nominal, operations,
        )?;
        if consumed_credit != 0 {
            self.emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?
                .release_storage(consumed_credit)?;
        }
        Ok(terminator)
    }
}
