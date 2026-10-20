use super::*;

pub(super) fn frame(bytes: &[u8], s: &mut Scope<'_, '_>) -> Result<DecodedNativeCpuInputV1, Error> {
    let mut r = Reader {
        bytes,
        s,
        nodes: 0,
        statements: 0,
        atoms: 0,
    };
    require(r.take(8)? == MAGIC, "magic")?;
    require(r.take(2)? == 1u16.to_le_bytes(), "version")?;
    require(r.take(2)? == [0; 2], "flags")?;
    require(r.u32()? as usize == bytes.len(), "frame length")?;
    require(r.u8()? == 64, "pointer width")?;
    r.s.reserve(size_of::<DecodedNativeCpuInputV1>())?;
    let semantic_mir_sha256 = r.array()?;
    let semantic_root = r.u32()?;
    require(
        u64::from(semantic_root) < HARD_MAX_FUNCTIONS_V1,
        "semantic root",
    )?;
    let registration_path = r.text()?;
    let subject = r.subject()?;
    Ok(DecodedNativeCpuInputV1 {
        semantic_mir_sha256,
        semantic_root,
        registration_path,
        logical_kernel_name: subject.logical_kernel_name,
        kernel: subject.kernel,
        reference: subject.reference,
        signature: subject.signature,
        effect_ir_sha256: subject.effect_ir_sha256,
        ir: subject.ir,
        commitment: [0; 32],
    })
}

pub(super) fn policy_frame(
    bytes: &[u8],
    s: &mut Scope<'_, '_>,
) -> Result<DecodedNativeCpuPolicyInputV2, Error> {
    let mut r = Reader {
        bytes,
        s,
        nodes: 0,
        statements: 0,
        atoms: 0,
    };
    require(r.take(8)? == policy_v2::POLICY_MAGIC, "magic")?;
    require(r.take(2)? == 2u16.to_le_bytes(), "version")?;
    require(r.take(2)? == [0; 2], "flags")?;
    require(r.u32()? as usize == bytes.len(), "frame length")?;
    require(r.u8()? == 64, "pointer width")?;
    r.s.reserve(size_of::<DecodedNativeCpuPolicyInputV2>())?;
    let semantic_mir_sha256 = r.array()?;
    let semantic_root = r.u32()?;
    require(
        u64::from(semantic_root) < HARD_MAX_FUNCTIONS_V1,
        "semantic root",
    )?;
    require(r.u8()? == policy_v2::ADMITTED_POLICY_TAG, "origin tag")?;
    require(
        r.take(2)? == policy_v2::REFERENCE_ENROLLMENT_DOMAIN_V1.to_le_bytes(),
        "enrollment domain",
    )?;
    let origin = ReferenceEnrollmentOriginV1 {
        rustc_invocation_sha256: r.array()?,
        native_policy_sha256: r.array()?,
        policy_generation: r.u64()?,
        mapping_ordinal: r.u32()?,
    };
    require(
        origin.mapping_ordinal < MAX_REFERENCE_ENROLLMENT_BINDINGS_V1,
        "mapping ordinal",
    )?;
    let subject = r.subject()?;
    Ok(DecodedNativeCpuPolicyInputV2 {
        semantic_mir_sha256,
        semantic_root,
        origin,
        subject,
        commitment: [0; 32],
    })
}

struct Reader<'a, 's, 'b, 'w> {
    bytes: &'a [u8],
    s: &'s mut Scope<'b, 'w>,
    nodes: usize,
    statements: usize,
    atoms: usize,
}
impl<'a> Reader<'a, '_, '_, '_> {
    fn subject(&mut self) -> Result<DecodedCpuSubject, Error> {
        let logical_kernel_name = self.text()?;
        let kernel = self.identity()?;
        let reference = self.identity()?;
        require(self.take(4)? == [0; 4], "signature header")?;
        let kernel_inputs = self.rows(MAX_REFERENCE_SIGNATURE_INPUTS_V1, 2, |r| {
            r.signature_input()
        })?;
        let reference_inputs = self.rows(MAX_REFERENCE_SIGNATURE_INPUTS_V1, 2, |r| {
            r.signature_input()
        })?;
        let signature = ReferenceLogicalSignaturePreimageV1::new(
            kernel_inputs,
            reference_inputs,
            ReferenceReturnShapeV1::Unit,
            SemanticExternAbiV1::Rust,
            SemanticFunctionSafetyV1::Safe,
            false,
        )?;
        let effect_ir_sha256 = self.array()?;
        let argument_count = self.u32()?;
        let local_count = self.u32()?;
        require(
            local_count > argument_count && u64::from(local_count) <= HARD_MAX_LOCALS_V1,
            "local count",
        )?;
        let relations = self.rows(MAX_REFERENCE_SIGNATURE_INPUTS_V1, 6, |r| r.relation())?;
        let blocks = self.rows(MAX_REFERENCE_BLOCKS_V1, 9, |r| r.block())?;
        require(self.u32()? == 0, "loop summaries unsupported")?;
        let observable_output_effects =
            self.rows(MAX_REFERENCE_STATEMENTS_V1, 22, |r| r.effect())?;
        require(self.bytes.is_empty(), "trailing bytes")?;
        Ok(DecodedCpuSubject {
            logical_kernel_name,
            kernel,
            reference,
            signature,
            effect_ir_sha256,
            ir: ReferenceEffectIrV1 {
                argument_count,
                local_count,
                relations,
                blocks,
                loop_summaries: Box::default(),
                observable_output_effects,
            },
        })
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Error> {
        require(count <= self.bytes.len(), "truncated frame")?;
        self.s.work(add(count, 1)?)?;
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.take(N)?
            .try_into()
            .map_err(|_| Error::Wire("truncated array"))
    }
    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.array::<1>()?[0])
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn u128(&mut self) -> Result<u128, Error> {
        Ok(u128::from_le_bytes(self.array()?))
    }
    fn boolean(&mut self) -> Result<bool, Error> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::Wire("boolean")),
        }
    }
    fn count(&mut self, max: usize, minimum_bytes: usize) -> Result<usize, Error> {
        let count = self.u32()? as usize;
        require(
            count <= max && count <= self.bytes.len() / minimum_bytes,
            "count or remaining bytes",
        )?;
        Ok(count)
    }
    fn rows<T>(
        &mut self,
        max: usize,
        minimum_bytes: usize,
        mut read: impl FnMut(&mut Self) -> Result<T, Error>,
    ) -> Result<Box<[T]>, Error> {
        let count = self.count(max, minimum_bytes)?;
        let mut rows = self.s.vector(count)?;
        for _ in 0..count {
            rows.push(read(self)?);
        }
        Ok(rows.into_boxed_slice())
    }
    fn text(&mut self) -> Result<String, Error> {
        let count = self.count(MAX_NATIVE_CPU_INPUT_BYTES_V1, 1)?;
        let bytes = self.take(count)?;
        let mut text = self.s.vector(count)?;
        text.extend_from_slice(bytes);
        self.s.work(count)?;
        String::from_utf8(text).map_err(|_| Error::Wire("UTF-8"))
    }
    fn enum_value<T: Copy>(&mut self, values: &[(u8, T)]) -> Result<T, Error> {
        let tag = self.u8()?;
        self.s.work(values.len())?;
        values
            .iter()
            .find_map(|(wire, value)| (*wire == tag).then_some(*value))
            .ok_or(Error::Wire("enum tag"))
    }
    fn scalar(&mut self) -> Result<ReferenceScalarTypeV1, Error> {
        self.enum_value(SCALARS)
    }
    fn identity(&mut self) -> Result<ReferenceFunctionIdentityV1, Error> {
        Ok(ReferenceFunctionIdentityV1 {
            def_path_hash: self.array()?,
            function_sha256: self.array()?,
            item_definition_sha256: self.array()?,
            monomorphization_sha256: self.array()?,
            generic_type_arguments_sha256: self.array()?,
            const_generic_arguments_sha256: self.array()?,
            rustc_mir_body_sha256: self.array()?,
        })
    }
    fn signature_input(&mut self) -> Result<ReferenceSignatureInputV1, Error> {
        Ok(match self.u8()? {
            0 => ReferenceSignatureInputV1::Scalar(self.scalar()?),
            1 => {
                let region = self.enum_value(&[
                    (0, ReferenceRegionV1::Erased),
                    (1, ReferenceRegionV1::Static),
                ])?;
                let mutability = self.enum_value(&[
                    (0, SemanticMutabilityV1::Immutable),
                    (1, SemanticMutabilityV1::Mutable),
                ])?;
                let tag = self.u8()?;
                let scalar = self.scalar()?;
                let pointee = match tag {
                    0 => ReferencePointeeV1::Scalar(scalar),
                    1 => ReferencePointeeV1::Slice(scalar),
                    _ => return Err(Error::Wire("pointee tag")),
                };
                ReferenceSignatureInputV1::Reference {
                    region,
                    mutability,
                    pointee,
                }
            }
            2 => {
                let carrier = self.enum_value(&[
                    (0, ReferenceCarrierV1::DisjointSlice),
                    (1, ReferenceCarrierV1::WriteOnlyDisjointSlice),
                ])?;
                ReferenceSignatureInputV1::NominalOutput {
                    carrier,
                    element: self.scalar()?,
                }
            }
            _ => return Err(Error::Wire("signature input tag")),
        })
    }
    fn relation(&mut self) -> Result<ReferenceArgumentRelationV1, Error> {
        let tag = self.u8()?;
        let argument = self.u32()?;
        if tag == 4 {
            return Ok(ReferenceArgumentRelationV1::PointCoordinate {
                reference_argument: argument,
                axis: self.u32()?,
            });
        }
        let scalar = self.scalar()?;
        Ok(match tag {
            0 => ReferenceArgumentRelationV1::ScalarInput { argument, scalar },
            1 => ReferenceArgumentRelationV1::SharedSliceInput {
                argument,
                element: scalar,
            },
            2 => ReferenceArgumentRelationV1::DisjointOutputSlice {
                argument,
                element: scalar,
            },
            3 => ReferenceArgumentRelationV1::DisjointOutputCoordinate {
                argument,
                element: scalar,
            },
            _ => return Err(Error::Wire("relation tag")),
        })
    }
    fn place(&mut self) -> Result<ReferencePlaceV1, Error> {
        Ok(ReferencePlaceV1 {
            local: self.u32()?,
            projection: self.rows(DEPTH, 1, Self::projection)?,
        })
    }
    fn projection(&mut self) -> Result<ReferencePlaceProjectionV1, Error> {
        Ok(match self.u8()? {
            0 => ReferencePlaceProjectionV1::Dereference,
            1 => ReferencePlaceProjectionV1::Field(self.u32()?),
            2 => ReferencePlaceProjectionV1::Index(self.u32()?),
            3 => ReferencePlaceProjectionV1::ConstantIndex {
                offset: self.u64()?,
                minimum_length: self.u64()?,
                from_end: self.boolean()?,
            },
            _ => return Err(Error::Wire("projection tag")),
        })
    }
    fn constant(&mut self) -> Result<ReferenceConstantV1, Error> {
        Ok(match self.u8()? {
            0 => ReferenceConstantV1::ZeroSized,
            1 => {
                let scalar = self.scalar()?;
                let bits = self.u128()?;
                scalar_bits(scalar, bits)?;
                ReferenceConstantV1::Scalar { scalar, bits }
            }
            _ => return Err(Error::Wire("constant tag")),
        })
    }
    fn operand(&mut self) -> Result<ReferenceOperandV1, Error> {
        Ok(match self.u8()? {
            0 => ReferenceOperandV1::Copy(self.place()?),
            1 => ReferenceOperandV1::Move(self.place()?),
            2 => ReferenceOperandV1::Constant(self.constant()?),
            _ => return Err(Error::Wire("operand tag")),
        })
    }
    fn value(&mut self) -> Result<ReferenceValueV1, Error> {
        Ok(match self.u8()? {
            0 => ReferenceValueV1::Use(self.operand()?),
            1 => {
                let operation = self.enum_value(BINARY)?;
                let checked = self.boolean()?;
                ReferenceValueV1::Binary {
                    operation,
                    checked,
                    lhs: self.operand()?,
                    rhs: self.operand()?,
                }
            }
            2 => ReferenceValueV1::Unary {
                operation: self.enum_value(UNARY)?,
                operand: self.operand()?,
            },
            3 => ReferenceValueV1::Cast {
                kind: self.enum_value(CASTS)?,
                source: self.scalar()?,
                target: self.scalar()?,
                operand: self.operand()?,
            },
            4 => ReferenceValueV1::InputLength {
                reference_argument: self.u32()?,
            },
            _ => return Err(Error::Wire("value tag (helpers unsupported)")),
        })
    }
    fn block(&mut self) -> Result<ReferenceBlockV1, Error> {
        let block = self.u32()?;
        let count = self.count(MAX_REFERENCE_STATEMENTS_V1, 15)?;
        self.statements = add(self.statements, count)?;
        require(
            self.statements <= MAX_REFERENCE_STATEMENTS_V1,
            "statement limit",
        )?;
        let mut assignments = self.s.vector(count)?;
        for _ in 0..count {
            assignments.push(ReferenceAssignmentV1 {
                statement: self.u32()?,
                destination: self.place()?,
                value: self.value()?,
            });
        }
        Ok(ReferenceBlockV1 {
            block,
            assignments: assignments.into_boxed_slice(),
            terminator: self.terminator()?,
        })
    }
    fn terminator(&mut self) -> Result<ReferenceTerminatorV1, Error> {
        Ok(match self.u8()? {
            0 => ReferenceTerminatorV1::Return,
            1 => ReferenceTerminatorV1::Goto {
                target: self.u32()?,
            },
            2 => ReferenceTerminatorV1::Switch {
                discriminant: self.operand()?,
                values: self.rows(MAX_REFERENCE_STATEMENTS_V1, 20, |r| {
                    Ok((r.u128()?, r.u32()?))
                })?,
                otherwise: self.u32()?,
            },
            3 => ReferenceTerminatorV1::Assert {
                condition: self.operand()?,
                expected: self.boolean()?,
                success: self.u32()?,
                bounds_check: if self.boolean()? {
                    Some(ReferenceBoundsCheckV1 {
                        index: self.operand()?,
                        length: self.operand()?,
                    })
                } else {
                    None
                },
            },
            _ => return Err(Error::Wire("terminator tag")),
        })
    }
    fn expr(&mut self) -> Result<ReferenceEffectExpressionV1, Error> {
        self.expression(1, &mut 0)
    }
    fn expression(
        &mut self,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<ReferenceEffectExpressionV1, Error> {
        *nodes = add(*nodes, 1)?;
        self.nodes = add(self.nodes, 1)?;
        require(
            depth <= DEPTH
                && *nodes <= MAX_REFERENCE_EXPRESSION_NODES_V1
                && self.nodes <= TOTAL_NODES,
            "expression depth/node limit",
        )?;
        self.s.work(1)?;
        Ok(match self.u8()? {
            0 => ReferenceEffectExpressionV1::PointCoordinate { axis: self.u32()? },
            1 => ReferenceEffectExpressionV1::KernelScalarArgument {
                argument: self.u32()?,
            },
            2 => ReferenceEffectExpressionV1::Constant(self.constant()?),
            3 => {
                let operation = self.enum_value(BINARY)?;
                let checked = self.boolean()?;
                let lhs = self.expression(depth + 1, nodes)?;
                let lhs = self.s.boxed(lhs)?;
                let rhs = self.expression(depth + 1, nodes)?;
                let rhs = self.s.boxed(rhs)?;
                ReferenceEffectExpressionV1::Binary {
                    operation,
                    checked,
                    lhs,
                    rhs,
                }
            }
            4 => {
                let operation = self.enum_value(UNARY)?;
                let operand = self.expression(depth + 1, nodes)?;
                ReferenceEffectExpressionV1::Unary {
                    operation,
                    operand: self.s.boxed(operand)?,
                }
            }
            5 => {
                let kind = self.enum_value(CASTS)?;
                let source = self.scalar()?;
                let target = self.scalar()?;
                let operand = self.expression(depth + 1, nodes)?;
                ReferenceEffectExpressionV1::Cast {
                    kind,
                    source,
                    target,
                    operand: self.s.boxed(operand)?,
                }
            }
            6 => {
                let reference_argument = self.u32()?;
                let index = self.expression(depth + 1, nodes)?;
                ReferenceEffectExpressionV1::InputLoad {
                    reference_argument,
                    index: self.s.boxed(index)?,
                }
            }
            7 => ReferenceEffectExpressionV1::InputLength {
                reference_argument: self.u32()?,
            },
            _ => return Err(Error::Wire("expression tag")),
        })
    }
    fn predicate(&mut self) -> Result<ReferencePathPredicateV1, Error> {
        self.atoms = 0;
        Ok(ReferencePathPredicateV1 {
            clauses: self.rows(MAX_REFERENCE_GUARD_CLAUSES_V1, 4, |r| {
                let count = r.count(MAX_REFERENCE_GUARD_ATOMS_V1, 4)?;
                r.atoms = add(r.atoms, count)?;
                require(r.atoms <= MAX_REFERENCE_GUARD_ATOMS_V1, "guard atom limit")?;
                let mut atoms = r.s.vector(count)?;
                for _ in 0..count {
                    atoms.push(r.atom()?);
                }
                Ok(ReferenceGuardClauseV1 {
                    atoms: atoms.into_boxed_slice(),
                })
            })?,
        })
    }
    fn atom(&mut self) -> Result<ReferenceGuardAtomV1, Error> {
        Ok(match self.u8()? {
            0 => ReferenceGuardAtomV1::SwitchValueSet {
                discriminant: self.expr()?,
                values: self.rows(MAX_REFERENCE_STATEMENTS_V1, 16, Self::u128)?,
                inside_set: self.boolean()?,
            },
            1 => ReferenceGuardAtomV1::Assert {
                condition: self.expr()?,
                expected: self.boolean()?,
            },
            _ => return Err(Error::Wire("guard atom tag")),
        })
    }
    fn effect(&mut self) -> Result<ReferenceOutputWriteV1, Error> {
        let argument = self.u32()?;
        let block = self.u32()?;
        let statement = self.u32()?;
        let coordinate = match self.u8()? {
            0 => ReferenceOutputCoordinateV1::LogicalPoint(self.rows(
                MAX_REFERENCE_POINT_AXES_V1,
                2,
                Self::expr,
            )?),
            1 => ReferenceOutputCoordinateV1::SingleCoordinate,
            2 => ReferenceOutputCoordinateV1::Dynamic(self.expr()?),
            3 => ReferenceOutputCoordinateV1::Constant {
                offset: self.u64()?,
                minimum_length: self.u64()?,
                from_end: self.boolean()?,
            },
            _ => return Err(Error::Wire("coordinate tag")),
        };
        Ok(ReferenceOutputWriteV1 {
            argument,
            block,
            statement,
            coordinate,
            guard: self.predicate()?,
            rhs: self.expr()?,
            value: self.value()?,
        })
    }
}
