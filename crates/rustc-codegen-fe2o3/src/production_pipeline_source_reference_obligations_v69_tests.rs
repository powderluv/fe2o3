//! Inert structural fixtures are not authenticated compiler sources or proofs.
use super::*;
use crate::reference_effect_v1::*;
use fe2o3_mir_model::semantic_mir_v1::*;
use fe2o3_verifier::portable_reference_v1::signature::{
    ReferenceCarrierV1, ReferencePointeeV1, ReferenceRegionV1, ReferenceReturnShapeV1,
    ReferenceSignatureInputV1,
};

#[test]
fn source_reference_input_rows_account_complete_identity_copies_and_borrowed_headers() {
    type IdentityFields = ([u8; 16], [[u8; 32]; 6]);
    type ReplayFields<'a> = (&'a (), &'a (), [u8; 32], &'a [()]);
    type InputFields<'a> = (usize, IdentityFields, IdentityFields, ReplayFields<'a>);
    assert_eq!(
        size_of::<InputFields<'_>>(),
        size_of::<fe2o3_verifier::SourceScalarReferenceInputV69<'_>>()
    );
    assert_eq!(
        input_headers::<usize>().unwrap(),
        size_of::<Vec<InputFields<'_>>>()
            + 2 * size_of::<InputFields<'_>>()
            + align_of::<InputFields<'_>>()
            + size_of::<Result<usize, Error>>()
            + align_of::<Result<usize, Error>>()
            + 6 * size_of::<&()>()
    );
}

fn reference() -> Reference {
    let scalar = ReferenceScalarTypeV1::U32;
    let signature_preimage = ReferenceLogicalSignaturePreimageV1::new(
        vec![ReferenceSignatureInputV1::NominalOutput {
            carrier: ReferenceCarrierV1::DisjointSlice,
            element: scalar,
        }]
        .into_boxed_slice(),
        vec![
            ReferenceSignatureInputV1::Scalar(ReferenceScalarTypeV1::Usize),
            ReferenceSignatureInputV1::Reference {
                region: ReferenceRegionV1::Erased,
                mutability: SemanticMutabilityV1::Mutable,
                pointee: ReferencePointeeV1::Scalar(scalar),
            },
        ]
        .into_boxed_slice(),
        ReferenceReturnShapeV1::Unit,
        SemanticExternAbiV1::Rust,
        SemanticFunctionSafetyV1::Safe,
        false,
    )
    .unwrap();
    let derived = signature_preimage.derive_relations_v1().unwrap();
    let constant = ReferenceConstantV1::Scalar { scalar, bits: 17 };
    let value = ReferenceValueV1::Use(ReferenceOperandV1::Constant(constant.clone()));
    let write = ReferenceOutputWriteV1 {
        argument: 0,
        block: 0,
        statement: 0,
        coordinate: ReferenceOutputCoordinateV1::LogicalPoint(
            vec![ReferenceEffectExpressionV1::PointCoordinate { axis: 0 }].into_boxed_slice(),
        ),
        guard: ReferencePathPredicateV1::unconditional_v1(),
        rhs: ReferenceEffectExpressionV1::Constant(constant),
        value: value.clone(),
    };
    let effect_ir = ReferenceEffectIrV1 {
        argument_count: 2,
        local_count: 3,
        relations: (0..derived.len())
            .map(|raw| derived.relation_at_raw_argument_v1(raw as u32).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        blocks: vec![ReferenceBlockV1 {
            block: 0,
            assignments: vec![ReferenceAssignmentV1 {
                statement: 0,
                destination: ReferencePlaceV1 {
                    local: 2,
                    projection: vec![ReferencePlaceProjectionV1::Dereference].into_boxed_slice(),
                },
                value,
            }]
            .into_boxed_slice(),
            terminator: ReferenceTerminatorV1::Return,
        }]
        .into_boxed_slice(),
        loop_summaries: Box::default(),
        observable_output_effects: vec![write.clone()].into_boxed_slice(),
    };
    let identity = ReferenceFunctionIdentityV1 {
        def_path_hash: [1; 16],
        function_sha256: [2; 32],
        item_definition_sha256: [3; 32],
        monomorphization_sha256: [4; 32],
        generic_type_arguments_sha256: [5; 32],
        const_generic_arguments_sha256: [6; 32],
        rustc_mir_body_sha256: [7; 32],
    };
    Reference {
        origin: ReferenceBindingOriginV1::SourceRegistration("inert-v69".into()),
        logical_kernel_name: "inert-v69".into(),
        kernel: identity,
        reference: identity,
        signature_preimage,
        effect_ir_sha256: effect_ir.canonical_sha256_v1(),
        effect_ir,
        observable_output_writes: vec![write].into_boxed_slice(),
    }
}

fn function(reference: &Reference, arity: usize, returns: usize) -> SemanticFunctionDeclV1 {
    let ty = SemanticTypeIdV1::from_index(0);
    let source = SemanticSourceProvenanceV1::unavailable();
    let identity = SemanticFunctionIdentityV1::from_sha256(reference.kernel.function_sha256);
    let local_identity = crate::rustc_semantic_adapter_v1::rustc_local_identity_v1(
        identity,
        reference.kernel.rustc_mir_body_sha256,
        0,
    );
    // A normalization temporary has no raw ordinal. Its retained identity must
    // not be confused with the original Return local's raw index zero.
    let mut locals = vec![SemanticLocalDeclV1::new(
        SemanticLocalIdentityV1::from_sha256([0; 32]),
        ty,
        SemanticLocalRoleV1::Temporary,
        source,
    )];
    locals.extend((0..returns).map(|_| {
        SemanticLocalDeclV1::new(local_identity, ty, SemanticLocalRoleV1::Return, source)
    }));
    locals.sort_unstable_by_key(SemanticLocalDeclV1::identity);
    SemanticFunctionDeclV1::new(
        identity,
        SemanticFunctionRoleV1::KernelRoot,
        SemanticItemDefinitionIdentityV1::from_sha256(reference.kernel.item_definition_sha256),
        SemanticMonomorphizationIdentityV1::from_sha256(reference.kernel.monomorphization_sha256),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256(
            reference.kernel.generic_type_arguments_sha256,
        ),
        SemanticConstGenericArgumentsIdentityV1::from_sha256(
            reference.kernel.const_generic_arguments_sha256,
        ),
        source,
        SemanticFunctionAbiV1::new(
            SemanticAbiIdentityV1::from_sha256([20; 32]),
            SemanticLayoutIdentityV1::from_sha256([21; 32]),
            SemanticCanonAbiV1::Rust,
            false,
            false,
            (0..arity)
                .map(|_| SemanticAbiValueV1::new(ty, SemanticAbiPassModeV1::Ignore))
                .collect(),
            SemanticAbiValueV1::new(ty, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap(),
        locals,
        SemanticBlockIdV1::from_index(0),
        vec![
            SemanticBasicBlockV1::new(
                crate::rustc_semantic_adapter_v1::rustc_block_identity_v1(
                    identity,
                    reference.kernel.rustc_mir_body_sha256,
                    0,
                ),
                source,
                vec![],
                SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn reference_kernel_binding_checks_all_retained_identity_axes_and_return_body_commitment() {
    let reference = reference();
    let body = function(&reference, 1, 1);
    assert_ne!(body.locals()[0].role(), SemanticLocalRoleV1::Return);
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    check_function(&reference, &body, &mut budget).unwrap();
    for axis in 0..6 {
        let mut foreign = reference.clone();
        match axis {
            0 => foreign.kernel.function_sha256[0] ^= 1,
            1 => foreign.kernel.item_definition_sha256[0] ^= 1,
            2 => foreign.kernel.monomorphization_sha256[0] ^= 1,
            3 => foreign.kernel.generic_type_arguments_sha256[0] ^= 1,
            4 => foreign.kernel.const_generic_arguments_sha256[0] ^= 1,
            5 => foreign.kernel.rustc_mir_body_sha256[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(
            matches!(
                check_function(&foreign, &body, &mut budget),
                Err(Error::ReferenceObligations(
                    ReferenceObligationErrorV69::Binding(_)
                ))
            ),
            "axis {axis}"
        );
    }
    for (arity, returns) in [(0, 1), (2, 1), (1, 0), (1, 2)] {
        let foreign = function(&reference, arity, returns);
        assert!(matches!(
            check_function(&reference, &foreign, &mut budget),
            Err(Error::ReferenceObligations(
                ReferenceObligationErrorV69::Binding(_)
            ))
        ));
    }
}

#[test]
fn reference_kernel_binding_has_independent_exact_work_and_sticky_denial_boundaries() {
    let reference = reference();
    let body = function(&reference, 1, 1);
    // Five 32-byte function axes and four scalar checks, then one local-role
    // scan and the fixed rustc-local commitment transcript debit.
    let identity_work =
        (8 + b"fe2o3/semantic-mir/rustc-local/v1".len()) + (8 + 32) + (8 + 32) + (8 + 4) + 32;
    let expected = 5 * 32 + 4 + body.locals().len() + identity_work;
    for limit in [expected - 1, expected] {
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, 37);
        budget.reserve_storage(37).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let result = check_function(&reference, &body, &mut budget);
        assert_eq!(result.is_ok(), limit == expected);
        assert_eq!(budget.storage(), 37);
        assert!(budget.work_ledger_identity_v1() == ledger);
        if limit < expected {
            assert!(matches!(result, Err(Error::Resource(Resource::Work(error)))
                if error.actual() == expected && error.limit() == limit));
            let prior = budget.check_prior_denials_v1().unwrap_err();
            let work = budget.work();
            assert!(matches!(check_function(&reference, &body, &mut budget),
                Err(Error::Resource(error)) if error == prior));
            assert_eq!(budget.work(), work);
        } else {
            assert_eq!(budget.work(), expected);
        }
    }
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, 37);
    budget.reserve_storage(37).unwrap();
    let prior = budget.reserve_storage(1).unwrap_err();
    assert!(matches!(check_function(&reference, &body, &mut budget),
        Err(Error::Resource(error)) if error == prior));
    assert_eq!((budget.work(), budget.storage()), (0, 37));
}

#[test]
fn reference_obligation_headers_have_an_independent_borrowed_owner_shape() {
    #[allow(dead_code)]
    struct DigestShape {
        digest: sha2::Sha256,
        canonical_transcript: Option<Vec<u8>>,
    }
    #[allow(dead_code)]
    struct RootShape<'a> {
        ordinal: usize,
        semantic_root: SemanticFunctionIdV1,
        body: &'a SemanticFunctionDeclV1,
        descriptor: &'a crate::compiler_descriptor::TypedDescriptorRootV1,
    }
    #[allow(dead_code)]
    struct ObligationShape<'a> {
        root: usize,
        reference: &'a Reference,
    }
    #[allow(dead_code)]
    struct PendingShape<'a> {
        source: &'a ProductionSemanticSsaOwnerV1,
        bindings: &'a AuthenticatedProductionBindings,
        roots: Vec<RootShape<'a>>,
        obligations: Vec<ObligationShape<'a>>,
        slot: usize,
        ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
        required: usize,
    }
    type ResultShape = Result<[u8; 513], Error>;
    let expected = size_of::<PendingShape<'_>>()
        + align_of::<PendingShape<'_>>()
        + size_of::<Vec<RootShape<'_>>>()
        + size_of::<Vec<ObligationShape<'_>>>()
        + size_of::<Vec<bool>>()
        + size_of::<ResultShape>()
        + align_of::<ResultShape>()
        + size_of::<(
            &ProductionSemanticSsaOwnerV1,
            &AuthenticatedProductionBindings,
        )>()
        + size_of::<DigestShape>()
        + align_of::<DigestShape>()
        + size_of::<[&[u8]; 3]>()
        + size_of::<[u8; 4]>()
        + 3 * size_of::<[u8; 32]>();
    assert_eq!(size_of::<Root<'_>>(), size_of::<RootShape<'_>>());
    assert_eq!(
        size_of::<Obligation<'_>>(),
        size_of::<ObligationShape<'_>>()
    );
    assert_eq!(headers::<[u8; 513]>().unwrap(), expected);
    assert!(headers::<[u8; 513]>().unwrap() > headers::<()>().unwrap());
    assert_eq!(
        replay_headers::<[u8; 513]>().unwrap(),
        size_of::<(&PendingShape<'_>, usize)>()
            + size_of::<ResultShape>()
            + align_of::<ResultShape>()
            + size_of::<Result<ResultShape, ReferenceBindingErrorV1>>()
            + align_of::<Result<ResultShape, ReferenceBindingErrorV1>>()
    );
}

#[test]
fn independent_reference_replay_keeps_cpu_body_and_effect_substitutions_closed() {
    for mutation in 0..4 {
        let mut reference = reference();
        match mutation {
            0 => {}
            1 => reference.effect_ir_sha256[0] ^= 1,
            2 => reference.observable_output_writes[0].argument = 1,
            3 => {
                reference.effect_ir.blocks[0].assignments[0].value = ReferenceValueV1::Use(
                    ReferenceOperandV1::Constant(ReferenceConstantV1::Scalar {
                        scalar: ReferenceScalarTypeV1::U32,
                        bits: 18,
                    }),
                );
                reference.effect_ir_sha256 = reference.effect_ir.canonical_sha256_v1();
            }
            _ => unreachable!(),
        }
        let entered = std::cell::Cell::new(false);
        let mut work = Work::new(usize::MAX);
        let mut budget = Budget::new(&mut work, usize::MAX);
        budget.reserve_storage(37).unwrap();
        let ledger = budget.work_ledger_identity_v1();
        let result = reference.with_replayed_output_writes_v1(&mut budget, |replay, budget| {
            entered.set(true);
            assert_eq!(
                replay.writes.as_slice(),
                reference.observable_output_writes.as_ref()
            );
            assert!(!std::ptr::eq(
                replay.writes.as_ptr(),
                reference.observable_output_writes.as_ptr()
            ));
            assert!(budget.work_ledger_identity_v1() == ledger);
            assert!(budget.storage() > 37);
        });
        assert_eq!(result.is_ok(), mutation == 0);
        assert_eq!(entered.get(), mutation == 0);
        assert_eq!(budget.storage(), 37);
        assert!(budget.work_ledger_identity_v1() == ledger);
    }
}

#[test]
fn reference_return_commitment_work_matches_independent_length_prefixed_transcript() {
    use sha2::Digest as _;
    let function = SemanticFunctionIdentityV1::from_sha256([19; 32]);
    let mir = [23; 32];
    let raw = 0u32.to_le_bytes();
    let mut transcript = Vec::new();
    for field in [
        b"fe2o3/semantic-mir/rustc-local/v1".as_slice(),
        function.as_bytes(),
        &mir,
        &raw,
    ] {
        transcript.extend_from_slice(&(field.len() as u64).to_le_bytes());
        transcript.extend_from_slice(field);
    }
    assert_eq!(RETURN_IDENTITY_WORK, transcript.len() + 32);
    let expected: [u8; 32] = sha2::Sha256::digest(&transcript).into();
    assert_eq!(
        crate::rustc_semantic_adapter_v1::rustc_local_identity_v1(function, mir, 0).as_bytes(),
        &expected
    );
}

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReferenceObligationObservationV69 {
    pub(crate) source: [u8; 32],
    pub(crate) roots: usize,
    pub(crate) obligations: usize,
    pub(crate) work: usize,
    pub(crate) peak: usize,
    pub(crate) substitutions: usize,
    pub(crate) custody_checks: usize,
    pub(crate) exact_boundaries: bool,
    pub(crate) semantic_proof_still_required: bool,
}

fn observe(pending: &Unproved<'_>, budget: &mut Budget<'_>) -> Result<usize, Error> {
    for index in 0..pending.obligations.len() {
        pending.with_replayed(index, budget, |effects, _| {
            assert!(!effects.writes.is_empty());
            Ok(())
        })?;
    }
    Ok(pending.obligations.len())
}

#[cfg(test)]
impl<'tcx> ProductionCompilation<'tcx, CollectedRustStage<'tcx>> {
    /// Uses two independent genuine imports; no synthetic source owner enters
    /// the production binding continuation.
    pub(crate) fn reference_obligations_observation_for_test_v69(
        self,
        donor: ProductionCompilation<'tcx, CollectedRustStage<'tcx>>,
    ) -> Result<ReferenceObligationObservationV69, Error> {
        let mut actual = self
            .import_semantic_mir_with_profile_v29(ImportProfile::SourceOwnedV29)?
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?;
        let donor = donor
            .import_semantic_mir_with_profile_v29(ImportProfile::SourceOwnedV29)?
            .construct_semantic_middle_end()?
            .construct_semantic_ssa()?;
        let source = &actual.stage.semantic_ssa;
        assert_eq!(
            source.source_semantic_sha256(),
            donor.stage.semantic_ssa.source_semantic_sha256()
        );
        assert!(!std::ptr::eq(source, &donor.stage.semantic_ssa));
        let roots = source.source_semantic().roots().len();
        assert_eq!(roots, 2);
        let bindings = &actual.stage.bindings;
        assert_eq!(bindings.reference_effect_bindings.as_slice().len(), roots);
        let mut work = Work::new(500_000_000);
        let mut budget = Budget::new(&mut work, 64_000_000);
        budget.reserve_storage(37)?;
        let ledger = budget.work_ledger_identity_v1();
        let obligations = with_unproved(source, bindings, &mut budget, observe)?;
        assert_eq!(obligations, roots);
        let exact_work = budget.work();
        let exact_peak = budget.peak_storage();
        assert_eq!(budget.storage(), 37);
        assert!(budget.work_ledger_identity_v1() == ledger);

        for (work_limit, storage_limit, succeeds) in [
            (exact_work, exact_peak, true),
            (exact_work - 1, exact_peak, false),
            (exact_work, exact_peak - 1, false),
        ] {
            let mut work = Work::new(work_limit);
            let mut budget = Budget::new(&mut work, storage_limit);
            budget.reserve_storage(37)?;
            let ledger = budget.work_ledger_identity_v1();
            let result = with_unproved(source, bindings, &mut budget, observe);
            assert_eq!(result.is_ok(), succeeds, "{result:?}");
            assert_eq!(budget.storage(), 37);
            assert!(budget.work_ledger_identity_v1() == ledger);
            if succeeds {
                assert_eq!(budget.work(), exact_work);
                assert_eq!(budget.peak_storage(), exact_peak);
            } else {
                let prior = budget.check_prior_denials_v1().unwrap_err();
                assert!(matches!(result, Err(Error::Resource(error)) if error == prior));
                let used = budget.work();
                assert!(
                    matches!(with_unproved(source, bindings, &mut budget, observe),
                    Err(Error::Resource(error)) if error == prior)
                );
                assert_eq!((budget.work(), budget.storage()), (used, 37));
            }
        }

        let mut custody_checks = 0;
        with_unproved(source, bindings, &mut budget, |pending, budget| {
            for (foreign_source, foreign_bindings) in [
                (&donor.stage.semantic_ssa, bindings),
                (source, &donor.stage.bindings),
            ] {
                assert!(matches!(
                    pending.check(foreign_source, foreign_bindings, budget),
                    Err(Error::ReferenceObligations(
                        ReferenceObligationErrorV69::Binding(_)
                    ))
                ));
                custody_checks += 1;
            }
            let mut work = Work::new(500_000_000);
            let mut foreign = Budget::new(&mut work, 64_000_000);
            foreign.reserve_storage(budget.storage())?;
            assert!(matches!(
                pending.check(source, bindings, &mut foreign),
                Err(Error::Resource(Resource::Accounting))
            ));
            custody_checks += 1;
            // An inert copied accounting handle isolates slot identity from
            // ledger identity; no query can use its empty metadata roster.
            let other_slot = Unproved {
                source,
                bindings,
                roots: Vec::new(),
                obligations: Vec::new(),
                slot: pending.slot ^ 1,
                ledger: pending.ledger,
                required: pending.required,
            };
            assert!(matches!(
                other_slot.check(source, bindings, budget),
                Err(Error::Resource(Resource::Accounting))
            ));
            custody_checks += 1;
            Ok(())
        })?;
        assert_eq!(budget.storage(), 37);

        let undercut = with_unproved(source, bindings, &mut budget, |pending, budget| {
            budget.release_storage(1)?;
            pending.check(source, bindings, budget)
        });
        assert!(matches!(
            undercut,
            Err(Error::Resource(Resource::Accounting))
        ));
        assert_eq!(budget.storage(), 37);
        custody_checks += 1;

        let refused = with_unproved(source, bindings, &mut budget, |pending, budget| {
            pending.with_replayed(0, budget, |_, _| {
                Err::<(), _>(Error::Unsupported("v69 selected callback refusal"))
            })
        });
        assert!(matches!(
            refused,
            Err(Error::Unsupported("v69 selected callback refusal"))
        ));
        assert_eq!(budget.storage(), 37);
        custody_checks += 1;

        let unwind = catch_unwind(AssertUnwindSafe(|| {
            with_unproved::<()>(source, bindings, &mut budget, |pending, budget| {
                pending.with_replayed::<()>(0, budget, |_, _| std::panic::panic_any(0x6969_u32))
            })
        }));
        let payload = match unwind {
            Err(payload) => payload,
            Ok(result) => panic!("reference unwind returned {result:?}"),
        };
        assert_eq!(*payload.downcast::<u32>().unwrap(), 0x6969);
        assert_eq!(budget.storage(), 37);
        assert!(budget.work_ledger_identity_v1() == ledger);
        custody_checks += 1;
        assert!(matches!(
            require_discharged(source, bindings, &mut budget),
            Err(Error::Unsupported(
                "source-owned scalar reference obligations"
            ))
        ));
        assert_eq!(budget.storage(), 37);

        {
            let limit = 500_000_000;
            let mut work = Work::new(limit);
            let mut exhausted = Budget::new(&mut work, 64_000_000);
            exhausted.reserve_storage(37)?;
            let entered = std::cell::Cell::new(false);
            let result = with_unproved(source, bindings, &mut exhausted, |pending, budget| {
                let row = &pending.obligations[0];
                let body = pending.roots[row.root].body;
                // Leave fifteen units at the independent CPU replay's first
                // sixteen-unit charge, after all v69 metadata/body checks.
                let before_replay = 5
                    + replay_headers::<()>()?
                    + size_of::<&std::cell::Cell<bool>>()
                    + 2
                    + 5 * 32
                    + 4
                    + body.locals().len()
                    + (8 + b"fe2o3/semantic-mir/rustc-local/v1".len())
                    + (8 + 32)
                    + (8 + 32)
                    + (8 + 4)
                    + 32;
                budget.charge_work(limit - budget.work() - before_replay - 15)?;
                pending.with_replayed(0, budget, |_, _| {
                    entered.set(true);
                    Ok(())
                })
            });
            let denial = exhausted.check_prior_denials_v1().unwrap_err();
            assert!(!entered.get());
            assert!(matches!(denial, Resource::Work(error)
                if error.actual() == limit + 1 && error.limit() == limit));
            assert!(matches!(result, Err(Error::Resource(error)) if error == denial));
            assert_eq!(exhausted.storage(), 37);
            let used = exhausted.work();
            assert!(
                matches!(with_unproved(source, bindings, &mut exhausted, observe),
                Err(Error::Resource(error)) if error == denial)
            );
            assert_eq!((exhausted.work(), exhausted.storage()), (used, 37));
            custody_checks += 1;
        }

        let original = actual
            .stage
            .bindings
            .reference_effect_bindings
            .as_slice()
            .to_vec();
        let mut substitutions = 0;
        for mutation in 0..5 {
            let mut changed = original.clone();
            match mutation {
                0 => changed[0].logical_kernel_name.push_str("_absent"),
                1 => changed.push(changed[0].clone()),
                2 => changed[0].kernel.rustc_mir_body_sha256[0] ^= 1,
                3 => changed[0].effect_ir_sha256[0] ^= 1,
                4 => changed[0].observable_output_writes[0].argument = u32::MAX,
                _ => unreachable!(),
            }
            actual.stage.bindings.reference_effect_bindings =
                AuthenticatedReferenceEffectBindingsV1::new(changed);
            let result = with_unproved(source, &actual.stage.bindings, &mut budget, observe);
            assert!(
                matches!(result, Err(Error::ReferenceObligations(_))),
                "mutation {mutation}: {result:?}"
            );
            assert_eq!(budget.storage(), 37);
            substitutions += 1;
        }
        actual.stage.bindings.reference_effect_bindings =
            AuthenticatedReferenceEffectBindingsV1::new(original);
        actual.stage.bindings.typed_descriptor_roots.swap(0, 1);
        assert!(matches!(
            with_unproved(source, &actual.stage.bindings, &mut budget, observe),
            Err(Error::ReferenceObligations(
                ReferenceObligationErrorV69::Binding(_)
            ))
        ));
        assert_eq!(budget.storage(), 37);
        substitutions += 1;
        actual.stage.bindings.typed_descriptor_roots.swap(0, 1);
        assert_eq!(
            with_unproved(source, &actual.stage.bindings, &mut budget, observe)?,
            roots
        );
        assert_eq!(budget.storage(), 37);

        Ok(ReferenceObligationObservationV69 {
            source: *source.source_semantic_sha256(),
            roots,
            obligations,
            work: exact_work,
            peak: exact_peak,
            substitutions,
            custody_checks,
            exact_boundaries: true,
            semantic_proof_still_required: true,
        })
    }
}
