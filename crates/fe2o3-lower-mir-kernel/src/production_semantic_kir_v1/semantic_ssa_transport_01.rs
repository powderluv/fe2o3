#[derive(Clone, Debug)]
struct SemanticPromotedLocalV1 {
    semantic_type: SemanticTypeIdV1,
    transport_semantic_type: SemanticTypeIdV1,
    transport: SemanticPromotedTransportV1,
    kernel_types: Box<[Type]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticPromotedTransportV1 {
    Semantic(SemanticPromotedBindingV1),
    DirectParameter { parameter_local: u32 },
    CheckedWave64Reference(Wave64NominalReferenceDescriptorV1),
    Execution,
    SourceEnumTag,
}

#[derive(Clone, Copy)]
enum BorrowedTransportTypeV1<'a> {
    Existing(&'a Type),
    Pointer {
        pointee: &'a Type,
        address_space: AddressSpace,
        access: AccessMode,
    },
}

trait SemanticTransportVisitorV1 {
    type Error;

    fn node(&mut self) -> Result<(), Self::Error>;
    fn component(
        &mut self,
        value: ValueId,
        ty: BorrowedTransportTypeV1<'_>,
    ) -> Result<(), Self::Error>;
    fn invalid(detail: &'static str) -> Self::Error;

    fn equal_types(&mut self, left: &Type, right: &Type) -> Result<bool, Self::Error> {
        Ok(left == right)
    }
}

struct LegacyTransportCollectorV1<'a>(&'a mut Vec<(ValueId, Type)>);

impl SemanticTransportVisitorV1 for LegacyTransportCollectorV1<'_> {
    type Error = &'static str;

    fn node(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    fn component(
        &mut self,
        value: ValueId,
        ty: BorrowedTransportTypeV1<'_>,
    ) -> Result<(), Self::Error> {
        let ty = match ty {
            BorrowedTransportTypeV1::Existing(ty) => ty.clone(),
            BorrowedTransportTypeV1::Pointer {
                pointee,
                address_space,
                access,
            } => Type::pointer(pointee.clone(), address_space, access),
        };
        self.0.push((value, ty));
        Ok(())
    }

    fn invalid(detail: &'static str) -> Self::Error {
        detail
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticPromotedBindingV1 {
    Ordinary,
    MathContext,
    CollectiveContext,
    WorkgroupLdsScope,
    WorkgroupPipeline {
        pipeline: SemanticTypeIdV1,
        element: SemanticTypeIdV1,
        payload_binding: SemanticPipelinePayloadBindingV1,
        buffers: u32,
        elements: u64,
        prefetch_distance: u32,
        packed_bits: u32,
        alignment: u32,
    },
    MatrixContext,
    WaveLane {
        wave_width: u32,
    },
    DynamicLds {
        dynamic_lds: SemanticTypeIdV1,
        element_storage: SemanticTypeIdV1,
        elements: u32,
        byte_extent: u64,
        alignment: u32,
        producer_function: SemanticFunctionIdV1,
        producer_block: SemanticBlockIdV1,
    },
    WorkgroupCollectiveScratch {
        element: SemanticTypeIdV1,
    },
    MatrixFragment {
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
    },
    AccumulatorFragment {
        contract: SemanticMfmaAccumulatorContractV1,
    },
    IndexWitness {
        index_space: SemanticDisjointIndexSpaceV1,
        disjoint: bool,
        availability: Option<SemanticCapabilityAvailabilityV1>,
    },
    OptionIndexWitness {
        index_space: SemanticDisjointIndexSpaceV1,
        disjoint: bool,
        availability: SemanticOptionAvailabilityV1,
    },
    GridLeader {
        availability: SemanticCapabilityAvailabilityV1,
    },
    OptionGridLeader {
        availability: SemanticOptionAvailabilityV1,
    },
    ComponentWitness {
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticCapabilityAvailabilityV1,
    },
    OptionComponentWitness {
        index_space: SemanticDisjointIndexSpaceV1,
        availability: SemanticOptionAvailabilityV1,
    },
    OptionPointer {
        element: ScalarType,
        address_space: AddressSpace,
        access: AccessMode,
        availability: SemanticOptionAvailabilityV1,
    },
    Gfx950LdsTransposeTile {
        format: SemanticGfx950LdsTransposeFormatV1,
        state: SemanticGfx950LdsTransposeStateV1,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticGfx950LdsTransposeStateV1 {
    Uninitialized,
    Staged,
    Published,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SemanticPipelinePayloadBindingV1 {
    Ordinary,
    MatrixFragment {
        contract: SemanticMfmaOperandContractV1,
        storage_layout: SemanticMfmaStorageLayoutV1,
    },
    AccumulatorFragment {
        contract: SemanticMfmaAccumulatorContractV1,
    },
}

impl SemanticPipelinePayloadBindingV1 {
    fn from_promoted(binding: SemanticPromotedBindingV1) -> Option<Self> {
        match binding {
            SemanticPromotedBindingV1::Ordinary => Some(Self::Ordinary),
            SemanticPromotedBindingV1::MatrixFragment {
                contract,
                storage_layout,
            } => Some(Self::MatrixFragment {
                contract,
                storage_layout,
            }),
            SemanticPromotedBindingV1::AccumulatorFragment { contract } => {
                Some(Self::AccumulatorFragment { contract })
            }
            _ => None,
        }
    }

    const fn promoted(self) -> SemanticPromotedBindingV1 {
        match self {
            Self::Ordinary => SemanticPromotedBindingV1::Ordinary,
            Self::MatrixFragment {
                contract,
                storage_layout,
            } => SemanticPromotedBindingV1::MatrixFragment {
                contract,
                storage_layout,
            },
            Self::AccumulatorFragment { contract } => {
                SemanticPromotedBindingV1::AccumulatorFragment { contract }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SemanticCurrentWaveV1 {
    width: u32,
}

impl SemanticCurrentWaveV1 {
    const fn new(width: u32) -> Self {
        Self { width }
    }
}

impl SemanticPromotedBindingV1 {
    fn transport_types(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
    ) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
        self.transport_types_with_allocation_v29(
            types,
            semantic_type,
            &mut CompilerCarrierAllocationV29 {
                budget: None,
                representation: ExecutionCfgRepresentationV29::LegacyAbi,
            },
        )
    }

    fn transport_types_with_allocation_v29(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        allocation: &mut CompilerCarrierAllocationV29<'_>,
    ) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
        allocation.charge(1)?;
        let transport = match self {
            Self::Ordinary => allocation.ordinary_types(types, semantic_type)?,
            Self::MathContext
            | Self::CollectiveContext
            | Self::WorkgroupLdsScope
            | Self::MatrixContext => allocation.reserve(0)?,
            Self::WorkgroupPipeline {
                pipeline,
                packed_bits,
                ..
            } => {
                if semantic_type != pipeline {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "promoted workgroup pipeline semantic type changed",
                    ));
                }
                let packed = pipeline_packed_type_v1(packed_bits).ok_or_else(|| {
                    unsupported(
                        0,
                        None,
                        None,
                        "promoted workgroup pipeline packed type changed",
                    )
                })?;
                let pointer =
                    allocation.pointer(packed, AddressSpace::Workgroup, AccessMode::ReadWrite)?;
                allocation.fixed([pointer])?
            }
            Self::WaveLane { .. } => allocation.fixed([Type::Scalar(ScalarType::U32)])?,
            Self::DynamicLds {
                dynamic_lds,
                element_storage,
                ..
            } => {
                if semantic_type != dynamic_lds {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "promoted dynamic LDS semantic type changed",
                    ));
                }
                let element = lower_dynamic_lds_element_with_allocation_v29(
                    types,
                    element_storage,
                    allocation,
                )?;
                let pointer =
                    allocation.pointer(element, AddressSpace::Workgroup, AccessMode::ReadWrite)?;
                allocation.fixed([pointer, Type::INDEX, Type::INDEX])?
            }
            Self::WorkgroupCollectiveScratch { element } => {
                lower_workgroup_collective_scratch_with_allocation_v29(
                    types,
                    semantic_type,
                    element,
                    allocation,
                )?
            }
            Self::MatrixFragment { contract, .. } => match contract.profile {
                SemanticMfmaProfileV1::Bf16F32M16N16K16 => {
                    allocation.repeated(&Type::Scalar(ScalarType::Bf16), 4)?
                }
                SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                | SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128 => {
                    allocation.repeated(&Type::Scalar(ScalarType::U32), 8)?
                }
            },
            Self::AccumulatorFragment { contract } => match contract.profile {
                SemanticMfmaProfileV1::Bf16F32M16N16K16
                | SemanticMfmaProfileV1::Fp4E2M1F32M16N16K128
                | SemanticMfmaProfileV1::Fp8E4M3F32M16N16K128 => {
                    allocation.repeated(&Type::Scalar(ScalarType::F32), 4)?
                }
            },
            Self::IndexWitness { .. } | Self::ComponentWitness { .. } => {
                allocation.fixed([Type::INDEX])?
            }
            Self::OptionIndexWitness { .. } | Self::OptionComponentWitness { .. } => {
                allocation.fixed([Type::BOOL, Type::INDEX])?
            }
            Self::OptionPointer {
                element,
                address_space,
                access,
                ..
            } => {
                let pointer = allocation.pointer(Type::Scalar(element), address_space, access)?;
                allocation.fixed([Type::BOOL, pointer])?
            }
            Self::GridLeader { .. } => allocation.reserve(0)?,
            Self::OptionGridLeader { .. } => allocation.fixed([Type::BOOL])?,
            Self::Gfx950LdsTransposeTile { .. } => {
                let pointer = allocation.pointer(
                    Type::Scalar(ScalarType::U8),
                    AddressSpace::Workgroup,
                    AccessMode::ReadWrite,
                )?;
                allocation.fixed([pointer])?
            }
        };
        Ok(transport)
    }

    const fn current_wave(self) -> Option<SemanticCurrentWaveV1> {
        match self {
            Self::Ordinary
            | Self::MathContext
            | Self::CollectiveContext
            | Self::WorkgroupLdsScope
            | Self::MatrixContext
            | Self::WorkgroupPipeline { .. }
            | Self::DynamicLds { .. }
            | Self::WorkgroupCollectiveScratch { .. }
            | Self::IndexWitness { .. }
            | Self::OptionIndexWitness { .. }
            | Self::GridLeader { .. }
            | Self::OptionGridLeader { .. }
            | Self::ComponentWitness { .. }
            | Self::OptionComponentWitness { .. }
            | Self::OptionPointer { .. } => None,
            Self::WaveLane { wave_width } => Some(SemanticCurrentWaveV1::new(wave_width)),
            Self::MatrixFragment { contract, .. } => {
                Some(SemanticCurrentWaveV1::new(contract.wave_width))
            }
            Self::AccumulatorFragment { contract } => {
                Some(SemanticCurrentWaveV1::new(contract.wave_width))
            }
            Self::Gfx950LdsTransposeTile { .. } => Some(SemanticCurrentWaveV1::new(64)),
        }
    }

    fn transport_values(
        self,
        binding: &SemanticValueBindingV1,
    ) -> Result<Vec<(ValueId, Type)>, &'static str> {
        let mut values = Vec::new();
        self.visit_transport_values(binding, &mut LegacyTransportCollectorV1(&mut values))?;
        Ok(values)
    }

    fn visit_transport_values<V: SemanticTransportVisitorV1>(
        self,
        binding: &SemanticValueBindingV1,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        use BorrowedTransportTypeV1::{Existing, Pointer};
        visitor.node()?;
        match (self, binding) {
            (Self::Ordinary, binding) => binding.visit_values_v1(visitor),
            (Self::MathContext, SemanticValueBindingV1::MathContext)
            | (Self::CollectiveContext, SemanticValueBindingV1::CollectiveContext)
            | (Self::WorkgroupLdsScope, SemanticValueBindingV1::WorkgroupLdsScope)
            | (Self::MatrixContext, SemanticValueBindingV1::MatrixContext) => Ok(()),
            (Self::WaveLane { wave_width }, SemanticValueBindingV1::WaveLane { value, wave })
                if wave_width == wave.width =>
            {
                visitor.component(*value, Existing(&Type::Scalar(ScalarType::U32)))
            }
            (
                Self::WorkgroupPipeline {
                    pipeline,
                    element,
                    payload_binding,
                    buffers,
                    elements,
                    prefetch_distance,
                    packed_bits,
                    alignment,
                },
                SemanticValueBindingV1::WorkgroupPipeline {
                    storage,
                    pipeline: actual_pipeline,
                    element: actual_element,
                    payload_binding: actual_payload_binding,
                    packed_type,
                    buffers: actual_buffers,
                    elements: actual_elements,
                    prefetch_distance: actual_prefetch_distance,
                    alignment: actual_alignment,
                    ..
                },
            ) if pipeline == *actual_pipeline
                && element == *actual_element
                && payload_binding.promoted() == *actual_payload_binding
                && buffers == *actual_buffers
                && elements == *actual_elements
                && prefetch_distance == *actual_prefetch_distance
                && alignment == *actual_alignment
                && pipeline_scalar_bit_width_v1(packed_type) == Some(packed_bits) =>
            {
                visitor.component(
                    *storage,
                    Pointer {
                        pointee: packed_type,
                        address_space: AddressSpace::Workgroup,
                        access: AccessMode::ReadWrite,
                    },
                )
            }
            (
                Self::DynamicLds {
                    dynamic_lds,
                    element_storage,
                    elements,
                    byte_extent,
                    alignment,
                    producer_function,
                    producer_block,
                },
                SemanticValueBindingV1::DynamicLds {
                    base,
                    base_ty,
                    len,
                    byte_len,
                    dynamic_lds: actual_dynamic_lds,
                    element_storage: actual_element_storage,
                    elements: actual_elements,
                    byte_extent: actual_byte_extent,
                    alignment: actual_alignment,
                    producer_function: actual_producer_function,
                    producer_block: actual_producer_block,
                },
            ) if dynamic_lds == *actual_dynamic_lds
                && element_storage == *actual_element_storage
                && elements == *actual_elements
                && byte_extent == *actual_byte_extent
                && alignment == *actual_alignment
                && producer_function == *actual_producer_function
                && producer_block == *actual_producer_block =>
            {
                visitor.component(*base, Existing(base_ty))?;
                visitor.component(*len, Existing(&Type::INDEX))?;
                visitor.component(*byte_len, Existing(&Type::INDEX))
            }
            (Self::WorkgroupCollectiveScratch { .. }, SemanticValueBindingV1::Aggregate(_)) => {
                binding.visit_values_v1(visitor)
            }
            (
                Self::MatrixFragment {
                    contract,
                    storage_layout,
                },
                SemanticValueBindingV1::MatrixFragment {
                    values,
                    contract: actual_contract,
                    storage_layout: actual_storage_layout,
                    wave,
                },
            ) if contract == *actual_contract
                && storage_layout == *actual_storage_layout
                && self.current_wave() == Some(*wave) =>
            {
                for (value, ty) in values {
                    visitor.component(*value, Existing(ty))?;
                }
                Ok(())
            }
            (
                Self::AccumulatorFragment { contract },
                SemanticValueBindingV1::AccumulatorFragment {
                    values,
                    contract: actual_contract,
                    wave,
                },
            ) if contract == *actual_contract && self.current_wave() == Some(*wave) => {
                for (value, ty) in values {
                    visitor.component(*value, Existing(ty))?;
                }
                Ok(())
            }
            (
                Self::Gfx950LdsTransposeTile { format, state },
                SemanticValueBindingV1::Gfx950LdsTransposeTile {
                    storage,
                    format: actual_format,
                    state: actual_state,
                },
            ) if format == *actual_format && state == *actual_state => visitor.component(
                *storage,
                Pointer {
                    pointee: &Type::Scalar(ScalarType::U8),
                    address_space: AddressSpace::Workgroup,
                    access: AccessMode::ReadWrite,
                },
            ),
            (
                Self::ComponentWitness {
                    index_space,
                    availability,
                },
                SemanticValueBindingV1::ComponentWitness {
                    raw,
                    index_space: actual_index_space,
                    availability: actual_availability,
                },
            ) if index_space == *actual_index_space && availability == *actual_availability => {
                visitor.component(*raw, Existing(&Type::INDEX))
            }
            (
                Self::IndexWitness {
                    index_space,
                    disjoint,
                    availability,
                },
                SemanticValueBindingV1::IndexWitness {
                    id,
                    index_space: actual_index_space,
                    disjoint: actual_disjoint,
                    availability: actual_availability,
                },
            ) if index_space == *actual_index_space
                && disjoint == *actual_disjoint
                && availability == *actual_availability =>
            {
                visitor.component(*id, Existing(&Type::INDEX))
            }
            (
                Self::OptionIndexWitness {
                    index_space,
                    disjoint,
                    availability,
                },
                SemanticValueBindingV1::OptionIndexWitness {
                    present,
                    id,
                    index_space: actual_index_space,
                    disjoint: actual_disjoint,
                    availability: actual_availability,
                },
            ) if index_space == *actual_index_space
                && disjoint == *actual_disjoint
                && availability == *actual_availability =>
            {
                visitor.component(*present, Existing(&Type::BOOL))?;
                visitor.component(*id, Existing(&Type::INDEX))
            }
            (
                Self::GridLeader { availability },
                SemanticValueBindingV1::GridLeader {
                    availability: actual_availability,
                },
            ) if availability == *actual_availability => Ok(()),
            (
                Self::OptionGridLeader { availability },
                SemanticValueBindingV1::OptionGridLeader {
                    present,
                    availability: actual_availability,
                },
            ) if availability == *actual_availability => {
                visitor.component(*present, Existing(&Type::BOOL))
            }
            (
                Self::OptionComponentWitness {
                    index_space,
                    availability,
                },
                SemanticValueBindingV1::OptionComponentWitness {
                    present,
                    raw,
                    index_space: actual_index_space,
                    availability: actual_availability,
                },
            ) if index_space == *actual_index_space && availability == *actual_availability => {
                visitor.component(*present, Existing(&Type::BOOL))?;
                visitor.component(*raw, Existing(&Type::INDEX))
            }
            (
                Self::OptionPointer {
                    element,
                    address_space,
                    access,
                    availability,
                },
                SemanticValueBindingV1::OptionPointer {
                    present,
                    pointer,
                    pointer_ty,
                    availability: actual_availability,
                },
            ) if matches!(pointer_ty, Type::Pointer(pointer)
                if *pointer.pointee == Type::Scalar(element)
                    && pointer.address_space == address_space && pointer.access == access)
                && availability == *actual_availability =>
            {
                visitor.component(*present, Existing(&Type::BOOL))?;
                visitor.component(*pointer, Existing(pointer_ty))
            }
            (Self::MatrixFragment { .. }, _) => Err(V::invalid(
                "promoted matrix fragment lacks its authenticated producer metadata",
            )),
            (Self::AccumulatorFragment { .. }, _) => Err(V::invalid(
                "promoted accumulator fragment lacks its authenticated producer metadata",
            )),
            (Self::Gfx950LdsTransposeTile { .. }, _) => Err(V::invalid(
                "promoted gfx950 LDS transpose tile lacks its authenticated state",
            )),
            (Self::ComponentWitness { .. }, _) => Err(V::invalid(
                "promoted component witness lacks its authenticated availability",
            )),
            (Self::IndexWitness { .. }, _) => Err(V::invalid(
                "promoted index witness lacks its authenticated producer metadata",
            )),
            (Self::OptionIndexWitness { .. }, _) => Err(V::invalid(
                "promoted optional index witness lacks its authenticated producer metadata",
            )),
            (Self::GridLeader { .. }, _) => Err(V::invalid(
                "promoted grid leader lacks its authenticated availability",
            )),
            (Self::OptionGridLeader { .. }, _) => Err(V::invalid(
                "promoted optional grid leader lacks its authenticated availability",
            )),
            (Self::OptionComponentWitness { .. }, _) => Err(V::invalid(
                "promoted optional component witness lacks its authenticated producer metadata",
            )),
            (Self::OptionPointer { .. }, _) => Err(V::invalid(
                "promoted optional pointer lacks its authenticated producer contract",
            )),
            (Self::WorkgroupCollectiveScratch { .. }, _) => Err(V::invalid(
                "promoted workgroup scratch lacks its authenticated aggregate",
            )),
            (Self::DynamicLds { .. }, _) => Err(V::invalid(
                "promoted dynamic LDS lacks its exact compiler-issued allocation contract",
            )),
            (Self::WorkgroupPipeline { .. }, _) => Err(V::invalid(
                "promoted workgroup pipeline lacks its compiler-issued storage contract",
            )),
            (Self::MathContext, _) => Err(V::invalid(
                "promoted math context lacks compiler-issued authority",
            )),
            (Self::CollectiveContext, _) => Err(V::invalid(
                "promoted collective context lacks compiler-issued authority",
            )),
            (Self::WorkgroupLdsScope, _) => Err(V::invalid(
                "promoted workgroup LDS scope lacks compiler-issued authority",
            )),
            (Self::MatrixContext, _) => Err(V::invalid(
                "promoted matrix context lacks compiler-issued authority",
            )),
            (Self::WaveLane { .. }, _) => Err(V::invalid(
                "promoted wave lane lacks compiler-issued authority",
            )),
        }
    }

    fn binding_from_transport(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        values: &[ValueDef],
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.binding_from_transport_with_allocation_v29(
            types,
            semantic_type,
            values,
            &mut CompilerCarrierAllocationV29 {
                budget: None,
                representation: ExecutionCfgRepresentationV29::LegacyAbi,
            },
        )
    }

    fn binding_from_transport_with_allocation_v29(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        values: &[ValueDef],
        allocation: &mut CompilerCarrierAllocationV29<'_>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        allocation.headers::<SemanticValueBindingV1>()?;
        if matches!(self, Self::Ordinary) {
            return allocation.ordinary_binding(types, semantic_type, values, true);
        }
        let expected =
            self.transport_types_with_allocation_v29(types, semantic_type, allocation)?;
        allocation.charge(values.len() + 1)?;
        let mut same = values.len() == expected.len();
        for (actual, expected) in values.iter().zip(&expected) {
            same &= match allocation.budget.as_deref_mut() {
                Some(budget) => invocation_equal_types_v1(&actual.ty, expected, budget)?,
                None => &actual.ty == expected,
            };
        }
        if !same {
            return Err(unsupported(
                0,
                None,
                None,
                "typed fragment SSA component types changed",
            ));
        }
        let components = allocation.components(values)?;
        match self {
            Self::Ordinary => allocation.ordinary_binding(types, semantic_type, values, true),
            Self::MathContext => Ok(SemanticValueBindingV1::MathContext),
            Self::CollectiveContext => Ok(SemanticValueBindingV1::CollectiveContext),
            Self::WorkgroupLdsScope => Ok(SemanticValueBindingV1::WorkgroupLdsScope),
            Self::WorkgroupPipeline {
                pipeline,
                element,
                payload_binding,
                buffers,
                elements,
                prefetch_distance,
                packed_bits,
                alignment,
            } => {
                let payload_binding = payload_binding.promoted();
                let component_types = payload_binding
                    .transport_types_with_allocation_v29(types, element, allocation)?
                    .into_boxed_slice();
                let packed_type = pipeline_packed_type_v1(packed_bits).ok_or_else(|| {
                    unsupported(
                        0,
                        None,
                        None,
                        "promoted workgroup pipeline packed type changed",
                    )
                })?;
                Ok(SemanticValueBindingV1::WorkgroupPipeline {
                    storage: components[0].0,
                    pipeline,
                    element,
                    payload_binding,
                    component_types,
                    packed_type,
                    buffers,
                    elements,
                    prefetch_distance,
                    alignment,
                })
            }
            Self::MatrixContext => Ok(SemanticValueBindingV1::MatrixContext),
            Self::WaveLane { wave_width } => Ok(SemanticValueBindingV1::WaveLane {
                value: components[0].0,
                wave: SemanticCurrentWaveV1::new(wave_width),
            }),
            Self::DynamicLds {
                dynamic_lds,
                element_storage,
                elements,
                byte_extent,
                alignment,
                producer_function,
                producer_block,
            } => Ok(SemanticValueBindingV1::DynamicLds {
                base: components[0].0,
                base_ty: allocation.clone_type(&components[0].1)?,
                len: components[1].0,
                byte_len: components[2].0,
                dynamic_lds,
                element_storage,
                elements,
                byte_extent,
                alignment,
                producer_function,
                producer_block,
            }),
            Self::WorkgroupCollectiveScratch { .. } => {
                allocation.ordinary_binding(types, semantic_type, values, false)
            }
            Self::MatrixFragment {
                contract,
                storage_layout,
            } => Ok(SemanticValueBindingV1::MatrixFragment {
                values: components,
                contract,
                storage_layout,
                wave: self
                    .current_wave()
                    .expect("matrix fragments have a current-wave association"),
            }),
            Self::AccumulatorFragment { contract } => {
                Ok(SemanticValueBindingV1::AccumulatorFragment {
                    values: components,
                    contract,
                    wave: self
                        .current_wave()
                        .expect("accumulator fragments have a current-wave association"),
                })
            }
            Self::ComponentWitness {
                index_space,
                availability,
            } => Ok(SemanticValueBindingV1::ComponentWitness {
                raw: components[0].0,
                index_space,
                availability,
            }),
            Self::IndexWitness {
                index_space,
                disjoint,
                availability,
            } => Ok(SemanticValueBindingV1::IndexWitness {
                id: components[0].0,
                index_space,
                disjoint,
                availability,
            }),
            Self::OptionIndexWitness {
                index_space,
                disjoint,
                availability,
            } => Ok(SemanticValueBindingV1::OptionIndexWitness {
                present: components[0].0,
                id: components[1].0,
                index_space,
                disjoint,
                availability,
            }),
            Self::GridLeader { availability } => {
                Ok(SemanticValueBindingV1::GridLeader { availability })
            }
            Self::OptionGridLeader { availability } => {
                Ok(SemanticValueBindingV1::OptionGridLeader {
                    present: components[0].0,
                    availability,
                })
            }
            Self::OptionComponentWitness {
                index_space,
                availability,
            } => Ok(SemanticValueBindingV1::OptionComponentWitness {
                present: components[0].0,
                raw: components[1].0,
                index_space,
                availability,
            }),
            Self::OptionPointer {
                element,
                address_space,
                access,
                availability,
            } => Ok(SemanticValueBindingV1::OptionPointer {
                present: components[0].0,
                pointer: components[1].0,
                pointer_ty: allocation.pointer(Type::Scalar(element), address_space, access)?,
                availability,
            }),
            Self::Gfx950LdsTransposeTile { format, state } => {
                Ok(SemanticValueBindingV1::Gfx950LdsTransposeTile {
                    storage: components[0].0,
                    format,
                    state,
                })
            }
        }
    }
}

impl SemanticPromotedTransportV1 {
    const fn uses_structural_enum_transport(self) -> bool {
        matches!(
            self,
            Self::Semantic(SemanticPromotedBindingV1::Ordinary) | Self::SourceEnumTag
        )
    }

    fn transport_types(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        direct_parameters: &BTreeMap<u32, Type>,
    ) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
        self.transport_types_with_representation_v29(
            types,
            semantic_type,
            direct_parameters,
            ExecutionCfgRepresentationV29::LegacyAbi,
        )
    }

    fn transport_types_with_representation_v29(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        direct_parameters: &BTreeMap<u32, Type>,
        representation: ExecutionCfgRepresentationV29,
    ) -> Result<Vec<Type>, ProductionSemanticKirErrorV1> {
        match self {
            Self::Semantic(binding) => binding.transport_types_with_allocation_v29(
                types,
                semantic_type,
                &mut CompilerCarrierAllocationV29 {
                    budget: None,
                    representation,
                },
            ),
            Self::CheckedWave64Reference(_) => Err(unsupported(0,None,None,"checked Wave64 types require live source relation")),
            Self::Execution | Self::SourceEnumTag => Err(execution_cfg_error_v29()),
            Self::DirectParameter { parameter_local } => direct_parameters
                .get(&parameter_local)
                .cloned()
                .map(|ty| vec![ty])
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch),
        }
    }

    fn transport_values(
        self,
        binding: &SemanticValueBindingV1,
        expected: &[Type],
    ) -> Result<Vec<(ValueId, Type)>, &'static str> {
        let mut values = Vec::new();
        self.visit_transport_values(
            binding,
            expected,
            &mut LegacyTransportCollectorV1(&mut values),
        )?;
        Ok(values)
    }

    fn visit_transport_values<V: SemanticTransportVisitorV1>(
        self,
        binding: &SemanticValueBindingV1,
        expected: &[Type],
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        visitor.node()?;
        match self {
            Self::Semantic(semantic) => semantic.visit_transport_values(binding, visitor),
            Self::CheckedWave64Reference(_) => Err(V::invalid("checked Wave64 values require live source relation")),
            Self::Execution | Self::SourceEnumTag => Err(V::invalid(
                "execution transport requires its captured CFG state",
            )),
            Self::DirectParameter { .. } => match (binding, expected) {
                (SemanticValueBindingV1::Value { id, ty }, [expected]) => {
                    if !visitor.equal_types(ty, expected)? {
                        return Err(V::invalid(
                            "promoted direct parameter changed its authenticated ABI carrier",
                        ));
                    }
                    visitor.component(*id, BorrowedTransportTypeV1::Existing(ty))
                }
                _ => Err(V::invalid(
                    "promoted direct parameter changed its authenticated ABI carrier",
                )),
            },
        }
    }

    fn binding_from_transport(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        values: &[ValueDef],
        expected: &[Type],
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.binding_from_transport_with_representation_v29(
            types,
            semantic_type,
            values,
            expected,
            ExecutionCfgRepresentationV29::LegacyAbi,
        )
    }

    fn binding_from_transport_with_representation_v29(
        self,
        types: &[SemanticTypeDeclV1],
        semantic_type: SemanticTypeIdV1,
        values: &[ValueDef],
        expected: &[Type],
        representation: ExecutionCfgRepresentationV29,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        match self {
            Self::Semantic(semantic) => semantic.binding_from_transport_with_allocation_v29(
                types,
                semantic_type,
                values,
                &mut CompilerCarrierAllocationV29 {
                    budget: None,
                    representation,
                },
            ),
            Self::CheckedWave64Reference(_) => Err(unsupported(0,None,None,"checked Wave64 reconstruction requires live source relation")),
            Self::Execution | Self::SourceEnumTag => Err(execution_cfg_error_v29()),
            Self::DirectParameter { .. }
                if values.len() == 1 && expected.len() == 1 && values[0].ty == expected[0] =>
            {
                Ok(SemanticValueBindingV1::Value {
                    id: values[0].id,
                    ty: values[0].ty.clone(),
                })
            }
            Self::DirectParameter { .. } => Err(unsupported(
                0,
                None,
                None,
                "promoted direct parameter changed its authenticated ABI carrier",
            )),
        }
    }
}

fn gfx950_lds_transpose_pointer_type_v1() -> Type {
    Type::pointer(
        Type::Scalar(ScalarType::U8),
        AddressSpace::Workgroup,
        AccessMode::ReadWrite,
    )
}

fn insert_compiler_issued_ssa_binding_v1(
    bindings: &mut BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    ty: SemanticTypeIdV1,
    binding: SemanticPromotedBindingV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if let Some(existing) = bindings.insert(ty, binding)
        && existing != binding
    {
        return Err(unsupported(
            0,
            None,
            None,
            "one semantic fragment type has conflicting compiler-issued contracts",
        ));
    }
    Ok(())
}
