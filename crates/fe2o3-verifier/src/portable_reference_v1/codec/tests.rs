//! Inert fixtures only; codec roundtrips confer no source/proof authority.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

#[path = "hostile_tests.rs"]
mod hostile;
#[path = "resource_tests.rs"]
mod resources;
#[path = "roundtrip_tests.rs"]
mod roundtrips;

#[path = "policy_tests.rs"]
mod policy;

struct Fixture {
    signature: ReferenceLogicalSignaturePreimageV1,
    ir: ReferenceEffectIrV1,
    writes: Box<[ReferenceOutputWriteV1]>,
    kernel: ReferenceFunctionIdentityV1,
    reference: ReferenceFunctionIdentityV1,
    digest: [u8; 32],
}
impl Fixture {
    fn input(&self) -> NativeCpuInputV1<'_> {
        NativeCpuInputV1 {
            association: NativeCpuAssociationV1 {
                semantic_mir_sha256: [19; 32],
                semantic_root: 2,
                registration_path: "fixture::register",
                logical_kernel_name: "fill",
            },
            kernel: &self.kernel,
            reference: &self.reference,
            replay: ReferenceReplayInputV1 {
                signature_preimage: &self.signature,
                effect_ir: &self.ir,
                effect_ir_sha256: self.digest,
                observable_output_writes: &self.writes,
            },
        }
    }
    fn refresh(&mut self) {
        self.writes = self.ir.observable_output_effects.clone();
        self.digest = self.ir.canonical_sha256_v1();
    }
}
fn identity(byte: u8) -> ReferenceFunctionIdentityV1 {
    ReferenceFunctionIdentityV1 {
        def_path_hash: [byte; 16],
        function_sha256: [byte; 32],
        item_definition_sha256: [byte; 32],
        monomorphization_sha256: [byte; 32],
        generic_type_arguments_sha256: [byte; 32],
        const_generic_arguments_sha256: [byte; 32],
        rustc_mir_body_sha256: [byte; 32],
    }
}
fn fixture() -> Fixture {
    let scalar = ReferenceScalarTypeV1::F32;
    let signature = ReferenceLogicalSignaturePreimageV1::new(
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
    let derived = signature.derive_relations_v1().unwrap();
    let relations = (0..derived.len())
        .map(|raw| derived.relation_at_raw_argument_v1(raw as u32).unwrap())
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let constant = ReferenceConstantV1::Scalar {
        scalar,
        bits: 0x422a_0000,
    };
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
    let ir = ReferenceEffectIrV1 {
        argument_count: 2,
        local_count: 3,
        relations,
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
    Fixture {
        signature,
        digest: ir.canonical_sha256_v1(),
        ir,
        writes: vec![write].into_boxed_slice(),
        kernel: identity(11),
        reference: identity(17),
    }
}

fn encode(input: NativeCpuInputV1<'_>) -> Result<(Vec<u8>, [u8; 32]), Error> {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    // The returned copy is test scaffold, not a production retained owner.
    with_encoded_native_cpu_input_v1(input, &mut budget, |bytes, hash, _| (bytes.to_vec(), hash))
}
fn decode(bytes: &[u8]) -> Result<(), Error> {
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    let result = with_decoded_native_cpu_input_v1(bytes, &mut budget, |_, _| ());
    assert_eq!(budget.storage(), 0);
    result
}

#[test]
fn roundtrip_is_deterministic_single_list_and_shared_replay_usable() {
    let f = fixture();
    let (bytes, hash) = encode(f.input()).unwrap();
    assert_eq!(encode(f.input()).unwrap(), (bytes.clone(), hash));
    assert_eq!(&bytes[..12], b"F2CPU1\0\0\x01\0\0\0");
    assert_eq!(
        u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize,
        bytes.len()
    );
    assert_eq!(bytes[16], 64);
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(&bytes);
    assert_eq!(hash, <[u8; 32]>::from(hasher.finalize()));
    let mut work = Work::new(usize::MAX);
    let mut budget = Budget::new(&mut work, usize::MAX);
    budget.reserve_storage(31).unwrap();
    with_decoded_native_cpu_input_v1(&bytes, &mut budget, |owner, budget| {
        let input = owner.input_v1();
        assert_eq!(owner.commitment_v1(), hash);
        assert_eq!(input.kernel, &f.kernel);
        assert_eq!(input.reference, &f.reference);
        assert_eq!(input.association.registration_path, "fixture::register");
        assert_eq!(input.association.semantic_root, 2);
        assert_eq!(input.replay.signature_preimage, &f.signature);
        assert_eq!(input.replay.effect_ir, &f.ir);
        assert!(std::ptr::eq(
            input.replay.observable_output_writes,
            input.replay.effect_ir.observable_output_effects.as_ref()
        ));
        with_replayed_output_writes_v1(input.replay, budget, |effects, _| {
            assert_eq!(effects.writes.as_slice(), f.writes.as_ref());
        })
        .unwrap();
        budget.reserve_storage(7).unwrap();
    })
    .unwrap();
    assert_eq!(budget.storage(), 38);
}

#[test]
fn every_truncation_and_trailing_byte_reject() {
    let (bytes, _) = encode(fixture().input()).unwrap();
    for end in 0..bytes.len() {
        assert!(decode(&bytes[..end]).is_err(), "prefix {end}");
    }
    let mut extra = bytes;
    extra.push(0);
    let length = extra.len() as u32;
    extra[12..16].copy_from_slice(&length.to_le_bytes());
    assert!(decode(&extra).is_err());
}

#[test]
fn retained_list_mismatch_and_legacy_digest_reject() {
    let mut f = fixture();
    f.writes[0].rhs = ReferenceEffectExpressionV1::Constant(ReferenceConstantV1::ZeroSized);
    assert!(encode(f.input()).is_err());
    let mut f = fixture();
    f.digest[0] ^= 1;
    assert_eq!(
        encode(f.input()).unwrap_err(),
        Error::Wire("legacy effect digest")
    );
}
