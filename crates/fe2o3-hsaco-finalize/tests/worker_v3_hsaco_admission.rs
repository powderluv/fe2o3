#![cfg(target_os = "linux")]

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use fe2o3_artifact_transaction::{
    AttemptScopedHsacoPublicationOutcomeV3, BuildInvocation, BuildSession,
    CompilerModuleHandoffReceiptV3, CompilerModuleHandoffSlotV3, ConsumedCompilerModuleHandoffV3,
    DurablePublishedHsacoClaimV3, ProducerIdentity, WorkerV3ExternalProviderPayloadsV1,
    WorkerV3PublicationIntentOutcomeV1, begin_build_attempt,
    consume_compiler_module_handoff_in_slot_v3, finish_build_attempt,
    publish_compiler_module_handoff_in_slot_v3, reacquire_current_hsaco_publication_lease_v3,
};
use fe2o3_compiler_ffi::{
    CodeObjectVersion, CompilerDescriptorSourceV1, CompilerFfiEnvelopeV1, CompilerModuleHandoffV2,
    CompilerModuleKindV1, CompilerModuleSymbolManifestV1, CompilerModuleSymbolRoleV1,
    INERT_COMPILER_MODULE_PAIR_BINDING_BYTES_V3, INERT_COMPILER_MODULE_PAIR_BINDING_MAGIC_V3,
    INERT_COMPILER_MODULE_PAIR_BINDING_VERSION_V3, INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_MAGIC_V3,
    INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_VERSION_V3, InertFinalCompilerModuleCommitmentV3,
    InertSemanticCompilerModuleHandoffV3,
};
use fe2o3_compiler_lineage::{
    InertLineageContentIdentityV3, InertProofBindingAssociationInputsV4,
    InertProofBindingAssociationV4,
};
use fe2o3_hsaco_finalize::{
    CompilerClosureV2, ContentIdentityV1, InertProtectedFirstBuildWorkerV3EvidenceV1,
    InspectedProtectedWorkerV3HsacoV1, LinkOptionV1, PinnedWorkerV1,
    ProtectedWorkerV3CompactFinalizerReplayV2, WorkerExecutionLimitsV1, WorkerInputKindV1,
    WorkerInputV1, WorkerMeasurementV1, WorkerOutputConstraintsV1, WorkerV3HsacoFinalizationError,
    WorkerV3HsacoInspectionError, WorkerV3HsacoPublicationErrorV1,
    execute_protected_reproducible_first_build_worker_v3, finalize_protected_worker_v3_hsaco_v1,
    inspect_protected_worker_v3_hsaco_v1, inspect_unfinalized,
    persist_prepared_protected_worker_v3_hsaco_publication_v1,
    prepare_protected_worker_v3_hsaco_publication_v1,
    publish_recovered_protected_worker_v3_hsaco_v1,
    recover_protected_worker_v3_hsaco_publication_v1,
};
use fe2o3_kernel_descriptor::{
    BlockSizeV1, BuildEvidenceV1, CanonicalCodeObjectDigest,
    CodeObjectVersion as DescriptorCodeObjectVersion, CompilerIdentityV1, DeviceDescriptorTableV1,
    DeviceLayoutDescriptorV1, DeviceLayoutRecordV1, DeviceTargetV1, DimensionsV1, EvidenceDigest,
    EvidenceIdentity, KernelAbiLayoutV1, KernelDescriptorV1, KernelId, LaunchConstraintsV1,
    LogicalArgumentV1, ProducerIdentityV1, ScalarTypeV1, SourceTypeDescriptorV1,
    SourceTypeRecordV1, Text, ValidName, encode_device_descriptor_table_v1,
};
use sha2::{Digest, Sha256};

#[path = "../../../tests/support/compiler_proof_inputs_v3.rs"]
mod compiler_proof_inputs_v3;
#[path = "fixtures/worker_v3_hsaco_test_support.rs"]
mod hsaco_fixture;

use compiler_proof_inputs_v3::{
    canonical_compiler_proof_inputs_v4, canonical_verus_execution_evidence_v1,
};
use hsaco_fixture::{
    ScalarAddFixtureMutation, scalar_add_fixture_with, slice_fixture_with_descriptor_table,
    slice_fixture_with_descriptor_table_and_workgroup,
    synthetic_two_kernel_slice_fixture_with_descriptor_table,
};

const TARGET: &str = "gfx942:xnack-";
const WORKER_BUILD_ID: &str = "fixture-worker-v3-hsaco-v1";
const RAW_HSACO_MARKER: &[u8] = b"; FE2O3/TEST-HSACO-PAYLOAD/V2-HEX:";
const CAPSULE_IDENTITY_DOMAIN_V3: &[u8] = b"FE2O3/INERT-PRODUCTION-SEMANTIC-CAPSULE/V3\0";
const PAIR_IDENTITY_DOMAIN_V3: &[u8] = b"FE2O3/INERT-COMPILER-MODULE-PAIR-BINDING/V3\0";
const OUTER_IDENTITY_DOMAIN_V3: &[u8] = b"FE2O3/INERT-SEMANTIC-COMPILER-MODULE-HANDOFF/V3\0";
const INVOCATION_DIGEST_DOMAIN_V3: &[u8] = b"FE2O3/RUSTC-BUILD-INVOCATION/V3\0";
const CAPSULE_MAGIC_V3: &[u8; 8] = b"F2O3ISV3";
const CAPSULE_VERSION_V3: u16 = 3;

const RECEIPTS: [(&str, &[u8]); 14] = [
    (
        "inventory",
        b"FE2O3/INERT-LINEAGE-CONTENT/RUSTC-IDENTITY-INVENTORY/V3\0",
    ),
    (
        "preflight",
        b"FE2O3/INERT-LINEAGE-CONTENT/RUSTC-PREFLIGHT-PLAN/V3\0",
    ),
    (
        "mir",
        b"FE2O3/INERT-LINEAGE-CONTENT/CANONICAL-SEMANTIC-MIR/V3\0",
    ),
    (
        "middle",
        b"FE2O3/INERT-LINEAGE-CONTENT/MIDDLE-END-PASS-CHAIN/V3\0",
    ),
    (
        "kir",
        b"FE2O3/INERT-LINEAGE-CONTENT/CANONICAL-KERNEL-IR/V3\0",
    ),
    (
        "correspondence",
        b"FE2O3/INERT-LINEAGE-CONTENT/MIR-TO-KIR-CORRESPONDENCE/V3\0",
    ),
    (
        "memory",
        b"FE2O3/INERT-LINEAGE-CONTENT/FORMAL-MEMORY-OBLIGATIONS/V3\0",
    ),
    (
        "proof",
        b"FE2O3/INERT-LINEAGE-CONTENT/PROOF-BINDING-SET/V3\0",
    ),
    ("target", b"FE2O3/INERT-LINEAGE-CONTENT/TARGET-BINDING/V3\0"),
    (
        "layout",
        b"FE2O3/INERT-LINEAGE-CONTENT/TARGET-DATA-LAYOUT/V3\0",
    ),
    ("abi", b"FE2O3/INERT-LINEAGE-CONTENT/ABI/V3\0"),
    (
        "exports",
        b"FE2O3/INERT-LINEAGE-CONTENT/EXPORT-MANIFEST/V3\0",
    ),
    (
        "lowering",
        b"FE2O3/INERT-LINEAGE-CONTENT/AMDGPU-LOWERING/V3\0",
    ),
    (
        "semantic-llvm",
        b"FE2O3/INERT-LINEAGE-CONTENT/SEMANTIC-TO-LLVM/V3\0",
    ),
];
const FINAL_RECEIPT_DOMAIN_V3: &[u8] =
    b"FE2O3/INERT-LINEAGE-CONTENT/FINAL-COMPILER-MODULE-COMMITMENT/V3\0";
const INVOCATION_20_HEX: &str = "4645324f33524900030000007c02000000000000010021212121212121212121212121212121212121212121212121212121212121212222222222222222222222222222222222222222222222222222222222222222232323232323232323232323232323232323232323232323232323232323232324242424242424242424242424242424242424242424242424242424242424242525252525252525252525252525252525252525252525252525252525252525262626262626262626262626262626262626262626262626262626262626262624242424242424242424242424242424242424242424242424242424242424242626262626262626262626262626262626262626262626262626262626262626100000002f776f726b73706163652f6665326f3307000000100000002f6f70742f6665326f332f72757374630c0000002d2d63726174652d6e616d650c000000776f726b65725f76335f3230230000006372617465732f776f726b65722d76332d666978747572652f7372632f6c69622e7273100000002d2d63726174652d747970653d6c69620e0000002d2d65646974696f6e3d32303234360000002d5a636f646567656e2d6261636b656e643d2f6f70742f6665326f332f6c696272757374635f636f646567656e5f6665326f332e736f040000001500434152474f5f4346475f5441524745545f4152434806000000616d6467636e0f004645324f335f485341434f5f4449521d0000002f776f726b73706163652f6665326f332f7461726765742f6665326f330c004645324f335f5441524745540d0000006766783934323a786e61636b2d16004645324f335f5645524946595f4b45524e454c5f49520100000031";
const INVOCATION_40_HEX: &str = "4645324f33524900030000007c02000000000000010041414141414141414141414141414141414141414141414141414141414141414242424242424242424242424242424242424242424242424242424242424242434343434343434343434343434343434343434343434343434343434343434344444444444444444444444444444444444444444444444444444444444444444545454545454545454545454545454545454545454545454545454545454545464646464646464646464646464646464646464646464646464646464646464644444444444444444444444444444444444444444444444444444444444444444646464646464646464646464646464646464646464646464646464646464646100000002f776f726b73706163652f6665326f3307000000100000002f6f70742f6665326f332f72757374630c0000002d2d63726174652d6e616d650c000000776f726b65725f76335f3430230000006372617465732f776f726b65722d76332d666978747572652f7372632f6c69622e7273100000002d2d63726174652d747970653d6c69620e0000002d2d65646974696f6e3d32303234360000002d5a636f646567656e2d6261636b656e643d2f6f70742f6665326f332f6c696272757374635f636f646567656e5f6665326f332e736f040000001500434152474f5f4346475f5441524745545f4152434806000000616d6467636e0f004645324f335f485341434f5f4449521d0000002f776f726b73706163652f6665326f332f7461726765742f6665326f330c004645324f335f5441524745540d0000006766783934323a786e61636b2d16004645324f335f5645524946595f4b45524e454c5f49520100000031";

#[derive(Clone, Copy)]
struct EvidenceConfig {
    attempt_seed: u8,
    slot: CompilerModuleHandoffSlotV3,
    invocation_seed: u8,
    module_seed: u8,
    optimization: &'static str,
    llvm_build_identity: &'static str,
    lineage_mutation: DescriptorLineageMutation,
}

#[derive(Clone, Copy)]
enum DescriptorLineageMutation {
    Exact,
    DifferentCanonicalSource,
    DifferentExportManifest,
}

impl EvidenceConfig {
    const BASE: Self = Self {
        attempt_seed: 0x61,
        slot: CompilerModuleHandoffSlotV3::Production,
        invocation_seed: 0x20,
        module_seed: 0x11,
        optimization: "2",
        llvm_build_identity: "upstream-llvm-test-build-a",
        lineage_mutation: DescriptorLineageMutation::Exact,
    };
}

#[allow(dead_code)]
pub(crate) struct PublishedWorkerV3Fixture {
    pub(crate) directory: TestDirectory,
    pub(crate) producer: ProducerIdentity,
    pub(crate) attempt: fe2o3_artifact_transaction::BuildAttempt,
    pub(crate) published: fe2o3_hsaco_finalize::PublishedProtectedWorkerV3HsacoV1,
}

#[allow(dead_code)]
pub(crate) struct PublishedWorkerV3InDirectory {
    pub(crate) producer: ProducerIdentity,
    pub(crate) attempt: fe2o3_artifact_transaction::BuildAttempt,
    pub(crate) published: fe2o3_hsaco_finalize::PublishedProtectedWorkerV3HsacoV1,
}

#[allow(dead_code)]
pub(crate) fn published_worker_v3_fixture() -> PublishedWorkerV3Fixture {
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    published_worker_v3_fixture_from_raw_hsaco(fixture.bytes, "vecadd", "vecadd.kd")
}

#[allow(dead_code)]
/// Publishes a hand-authored two-entry fixture; this is not compiler-produced provenance.
pub(crate) fn published_synthetic_two_kernel_worker_v3_fixture() -> PublishedWorkerV3Fixture {
    let fixture = synthetic_two_kernel_slice_fixture_with_descriptor_table(
        &synthetic_two_kernel_slice_descriptor_table(),
    );
    published_worker_v3_fixture_from_raw_hsaco_for_kernels(
        fixture.bytes,
        &[
            ("synthetic_first_transform", "synthetic_first_transform.kd"),
            (
                "synthetic_second_transform",
                "synthetic_second_transform.kd",
            ),
        ],
    )
}

#[allow(dead_code)]
pub(crate) fn published_worker_v3_fixture_from_raw_hsaco(
    raw_hsaco: Vec<u8>,
    entry_symbol: &str,
    descriptor_symbol: &str,
) -> PublishedWorkerV3Fixture {
    published_worker_v3_fixture_from_raw_hsaco_for_kernels_with_config(
        raw_hsaco,
        &[(entry_symbol, descriptor_symbol)],
        EvidenceConfig::BASE,
    )
}

fn published_worker_v3_fixture_from_raw_hsaco_for_kernels(
    raw_hsaco: Vec<u8>,
    kernel_symbols: &[(&str, &str)],
) -> PublishedWorkerV3Fixture {
    published_worker_v3_fixture_from_raw_hsaco_for_kernels_with_config(
        raw_hsaco,
        kernel_symbols,
        EvidenceConfig::BASE,
    )
}

fn published_worker_v3_fixture_from_raw_hsaco_for_kernels_with_config(
    raw_hsaco: Vec<u8>,
    kernel_symbols: &[(&str, &str)],
    config: EvidenceConfig,
) -> PublishedWorkerV3Fixture {
    let directory = TestDirectory::new();
    let staged = publish_worker_v3_fixture_in_directory_for_kernels_with_config(
        &directory,
        raw_hsaco,
        kernel_symbols,
        config,
    );
    PublishedWorkerV3Fixture {
        directory,
        producer: staged.producer,
        attempt: staged.attempt,
        published: staged.published,
    }
}

#[allow(dead_code)]
pub(crate) fn publish_worker_v3_fixture_in_directory(
    directory: &TestDirectory,
    attempt_seed: u8,
) -> PublishedWorkerV3InDirectory {
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    publish_worker_v3_fixture_in_directory_for_kernels_with_config(
        directory,
        fixture.bytes,
        &[("vecadd", "vecadd.kd")],
        EvidenceConfig {
            attempt_seed,
            ..EvidenceConfig::BASE
        },
    )
}

fn publish_worker_v3_fixture_in_directory_for_kernels_with_config(
    directory: &TestDirectory,
    raw_hsaco: Vec<u8>,
    kernel_symbols: &[(&str, &str)],
    config: EvidenceConfig,
) -> PublishedWorkerV3InDirectory {
    let producer = producer();
    let (attempt, source) = evidence_in_directory_for_kernels_and_providers(
        directory,
        raw_hsaco,
        config,
        kernel_symbols,
        Vec::new(),
    );
    let inspected = inspect_protected_worker_v3_hsaco_v1(source).unwrap();
    let finalized = finalize_protected_worker_v3_hsaco_v1(inspected).unwrap();
    let prepared = prepare_protected_worker_v3_hsaco_publication_v1(&producer, finalized).unwrap();
    let persisted = persist_prepared_protected_worker_v3_hsaco_publication_v1(
        &directory.0,
        &producer,
        prepared,
    )
    .unwrap();
    let compiler_closure = persisted
        .finalized_evidence()
        .binding_expectation()
        .compiler_closure();
    let published = publish_recovered_protected_worker_v3_hsaco_v1(
        &directory.0,
        &producer,
        compiler_closure,
        persisted,
    )
    .unwrap();
    PublishedWorkerV3InDirectory {
        producer,
        attempt,
        published,
    }
}

#[test]
fn native_v3_inspection_retains_every_boundary_axis_without_authority() {
    let fixture = scalar_add_fixture_with(ScalarAddFixtureMutation::RequiredWorkgroup);
    let exact = fixture.bytes.clone();
    let evidence = evidence(fixture.bytes, EvidenceConfig::BASE);
    let source_identity = evidence.identity();
    let binding = evidence.binding();
    let expected = binding.expectation();
    let plan = evidence.plan().identity();

    let inspected = inspect_protected_worker_v3_hsaco_v1(evidence).unwrap();
    require_v3_inspection(&inspected);
    assert_eq!(inspected.source_evidence_identity(), source_identity);
    assert_eq!(inspected.binding_identity(), binding.identity());
    assert_eq!(inspected.binding_expectation(), expected);
    assert_eq!(inspected.attempt(), expected.attempt());
    assert_eq!(inspected.handoff_slot(), expected.slot());
    assert_eq!(
        inspected.transaction_identity(),
        expected.transaction_identity()
    );
    assert_eq!(
        inspected.outer_handoff_identity(),
        expected.outer_handoff_identity()
    );
    assert_eq!(
        inspected.outer_handoff().identity(),
        expected.outer_handoff_identity()
    );
    assert_eq!(inspected.compiler_closure(), expected.compiler_closure());
    assert_eq!(inspected.link_plan_identity(), plan);
    assert_eq!(inspected.exact_bytes(), exact);
    assert_eq!(
        inspected.raw_hsaco_identity(),
        ContentIdentityV1::calculate(&exact)
    );
    assert_eq!(
        inspected.linked_output_identity(),
        inspected.raw_hsaco_identity()
    );
    assert_eq!(inspected.target().to_string(), TARGET);
    assert_eq!(
        inspected.code_object_version(),
        fe2o3_kernel_descriptor::CodeObjectVersion::V6
    );
    assert_eq!(
        inspected.policy().launch().required_workgroup_size(),
        [64, 1, 1]
    );
    assert_eq!(inspected.policy().launch().wavefront_size(), 64);
    assert!(!inspected.descriptor_observation_preimage().is_empty());
    assert!(!inspected.abi_observation_preimage().is_empty());
    assert!(!inspected.resource_observation_preimage().is_empty());
    assert_eq!(inspected.source_evidence().identity(), source_identity);
    assert!(!inspected.canonical_descriptor_finalization_ran());
    assert!(!inspected.authenticates_compiler_origin());
    assert!(!inspected.proves_semantic_correctness());
    assert!(!inspected.grants_compiler_authority());
    assert!(!inspected.grants_link_authority());
    assert!(!inspected.grants_publication_authority());
    assert!(!inspected.grants_load_authority());
    assert!(!inspected.grants_launch_authority());
}

#[test]
fn strict_v3_inspection_derives_wg256_from_the_bound_descriptor() {
    let directory = TestDirectory::new();
    let descriptor = slice_descriptor_table_with_workgroup(256);
    let fixture = slice_fixture_with_descriptor_table_and_workgroup(&descriptor, 256);
    let (_, source) = evidence_in_directory_for_kernel(
        &directory,
        fixture.bytes,
        EvidenceConfig::BASE,
        "vecadd",
        "vecadd.kd",
    );

    let inspected = inspect_protected_worker_v3_hsaco_v1(source).unwrap();

    assert_eq!(
        inspected.policy().launch().required_workgroup_size(),
        [256, 1, 1]
    );
    assert_eq!(inspected.policy().launch().max_flat_workgroup_size(), 256);
}

#[test]
fn native_v3_finalization_fails_closed_without_descriptor_source_evidence() {
    let raw = inspected(
        scalar_add_fixture_with(ScalarAddFixtureMutation::RequiredWorkgroup).bytes,
        EvidenceConfig::BASE,
    );
    let raw_identity = raw.identity();
    let source_identity = raw.source_evidence_identity();
    let binding = raw.binding_identity();
    let expected = raw.binding_expectation();
    let raw_output = raw.raw_hsaco_identity();
    let blocker = match finalize_protected_worker_v3_hsaco_v1(raw) {
        Err(
            WorkerV3HsacoFinalizationError::MissingAuthenticatedProtectedDescriptorSourceEvidenceV3(
                blocker,
            ),
        ) => blocker,
        result => panic!("expected native V3 descriptor-source blocker, found {result:?}"),
    };

    assert_eq!(blocker.raw_inspection_identity(), raw_identity);
    assert_eq!(blocker.source_evidence_identity(), source_identity);
    assert_eq!(blocker.binding_identity(), binding);
    assert_eq!(blocker.binding_expectation(), expected);
    assert_eq!(blocker.attempt(), expected.attempt());
    assert_eq!(blocker.handoff_slot(), expected.slot());
    assert_eq!(
        blocker.transaction_identity(),
        expected.transaction_identity()
    );
    assert_eq!(
        blocker.outer_handoff_identity(),
        expected.outer_handoff_identity()
    );
    assert_eq!(blocker.compiler_closure(), expected.compiler_closure());
    assert_eq!(blocker.raw_output_identity(), raw_output);
    assert!(!blocker.may_infer_descriptor_claims_from_executable_metadata());
    assert!(!blocker.grants_publication_authority());
    assert!(!blocker.grants_load_authority());
    assert!(!blocker.grants_launch_authority());
}

#[test]
fn native_v3_finalization_rejects_a_different_canonical_descriptor_source() {
    let directory = TestDirectory::new();
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    let (_, source) = evidence_in_directory_for_kernel(
        &directory,
        fixture.bytes,
        EvidenceConfig {
            lineage_mutation: DescriptorLineageMutation::DifferentCanonicalSource,
            ..EvidenceConfig::BASE
        },
        "vecadd",
        "vecadd.kd",
    );
    let raw = inspect_protected_worker_v3_hsaco_v1(source).unwrap();
    assert!(matches!(
        finalize_protected_worker_v3_hsaco_v1(raw),
        Err(WorkerV3HsacoFinalizationError::CompilerDescriptorSourceMismatch)
    ));
}

#[test]
fn native_v3_finalization_rejects_a_different_export_manifest_receipt() {
    let directory = TestDirectory::new();
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    let (_, source) = evidence_in_directory_for_kernel(
        &directory,
        fixture.bytes,
        EvidenceConfig {
            lineage_mutation: DescriptorLineageMutation::DifferentExportManifest,
            ..EvidenceConfig::BASE
        },
        "vecadd",
        "vecadd.kd",
    );
    let raw = inspect_protected_worker_v3_hsaco_v1(source).unwrap();
    assert!(matches!(
        finalize_protected_worker_v3_hsaco_v1(raw),
        Err(WorkerV3HsacoFinalizationError::ExportManifestMismatch)
    ));
}

#[test]
fn native_v3_publication_persists_and_reconstructs_exact_lineage_after_restart() {
    let directory = TestDirectory::new();
    let config = EvidenceConfig::BASE;
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    let exact_raw = fixture.bytes.clone();
    let (attempt, source) = evidence_in_directory_for_kernel_and_providers(
        &directory,
        fixture.bytes,
        config,
        "vecadd",
        "vecadd.kd",
        Vec::new(),
    );
    let inspected = inspect_protected_worker_v3_hsaco_v1(source).unwrap();
    let finalized = finalize_protected_worker_v3_hsaco_v1(inspected).unwrap();
    let exact_finalized = finalized.exact_finalized_bytes().to_vec();
    let prepared =
        prepare_protected_worker_v3_hsaco_publication_v1(&producer(), finalized).unwrap();
    assert_eq!(prepared.attempt(), attempt);
    assert!(!prepared.grants_publication_authority());
    assert!(!prepared.grants_load_authority());
    assert!(!prepared.grants_launch_authority());

    let persisted = persist_prepared_protected_worker_v3_hsaco_publication_v1(
        &directory.0,
        &producer(),
        prepared,
    )
    .unwrap();
    assert_eq!(
        persisted.outcome(),
        WorkerV3PublicationIntentOutcomeV1::Persisted
    );
    let compiler_closure = persisted
        .finalized_evidence()
        .binding_expectation()
        .compiler_closure();
    let binding = persisted.publication_binding(compiler_closure).unwrap();
    assert_eq!(
        binding.publication_intent_record_identity(),
        persisted.storage_record().identity().as_bytes()
    );
    assert_eq!(
        binding.finalization_identity(),
        *persisted
            .publication_intent()
            .finalization_identity()
            .as_bytes()
    );
    assert_eq!(
        binding.finalized_output_length(),
        exact_finalized.len() as u64
    );
    assert!(!binding.grants_publication_authority());
    assert!(!binding.grants_load_authority());
    assert!(!binding.grants_launch_authority());
    let mismatched_closure =
        CompilerClosureV2::new([21; 32], [22; 32], [23; 32], [24; 32], [25; 32], [26; 32]).unwrap();
    assert!(matches!(
        persisted.publication_binding(mismatched_closure),
        Err(WorkerV3HsacoPublicationErrorV1::CompilerClosureMismatch)
    ));
    let published = publish_recovered_protected_worker_v3_hsaco_v1(
        &directory.0,
        &producer(),
        compiler_closure,
        persisted,
    )
    .unwrap();
    let publication = published.publication_result();
    assert_eq!(publication.publication_binding(), binding);
    let encoded_claim = publication.published_claim().encode_canonical().unwrap();
    let decoded_claim = DurablePublishedHsacoClaimV3::decode_canonical(&encoded_claim).unwrap();
    assert_eq!(&decoded_claim, publication.published_claim());
    let lease = reacquire_current_hsaco_publication_lease_v3(&directory.0, &decoded_claim).unwrap();
    assert_eq!(lease.exact_artifact_bytes(), exact_finalized);
    drop(lease);
    assert_eq!(
        published.recovered_evidence().exact_finalized_hsaco(),
        exact_finalized
    );
    assert_eq!(
        fe2o3_hsaco_finalize::derive_unfinalized_hsaco_from_finalized_v1(
            published.recovered_evidence().exact_finalized_hsaco()
        )
        .unwrap(),
        exact_raw
    );
    let compiler_subject = published.compiler_execution_subject_v1().unwrap();
    let finalized = published.recovered_evidence().finalized_evidence();
    assert_eq!(compiler_subject.attempt(), finalized.attempt());
    assert_eq!(compiler_subject.slot(), finalized.handoff_slot());
    assert_eq!(
        compiler_subject.transaction_identity(),
        finalized.transaction_identity()
    );
    assert_eq!(
        compiler_subject.outer_handoff().sha256(),
        finalized.outer_handoff().identity().sha256()
    );
    let expected_intent = published.recovered_evidence().publication_intent();
    drop(published);
    finish_build_attempt(&directory.0, &producer(), attempt).unwrap();

    let recovered =
        recover_protected_worker_v3_hsaco_publication_v1(&directory.0, &producer(), attempt)
            .unwrap();
    assert_eq!(
        recovered.outcome(),
        WorkerV3PublicationIntentOutcomeV1::Recovered
    );
    assert_eq!(recovered.exact_finalized_hsaco(), exact_finalized);
    assert_eq!(recovered.publication_intent(), expected_intent);
    assert_eq!(
        recovered.compiler_execution_subject_v1().unwrap(),
        compiler_subject
    );
    assert!(!recovered.grants_publication_authority());
    assert!(!recovered.grants_load_authority());
    assert!(!recovered.grants_launch_authority());

    let reconstructed = publish_recovered_protected_worker_v3_hsaco_v1(
        &directory.0,
        &producer(),
        compiler_closure,
        recovered,
    )
    .unwrap();
    assert_eq!(
        reconstructed.publication_result().outcome(),
        AttemptScopedHsacoPublicationOutcomeV3::RecoveredCommittedPublication
    );
    assert_eq!(
        reconstructed.recovered_evidence().exact_finalized_hsaco(),
        exact_finalized
    );
    assert_eq!(
        reconstructed.compiler_execution_subject_v1().unwrap(),
        compiler_subject
    );
    let binding = reconstructed.publication_result().publication_binding();
    let (replay, record, claim, lease) = reconstructed
        .into_load_envelope_parts_v1()
        .expect("completed V3 publication must transfer into load-envelope custody")
        .into_parts();
    assert_eq!(
        record.identity().as_bytes(),
        binding.publication_intent_record_identity()
    );
    assert_eq!(record.plan(), claim.plan());
    assert_eq!(claim.worker_v3_binding(), binding);
    assert_eq!(replay.finalized_hsaco, exact_finalized);
    assert_eq!(lease.exact_artifact_bytes(), exact_finalized);
    assert!(replay.external_provider_payloads.is_empty());
    let providers =
        WorkerV3ExternalProviderPayloadsV1::new(replay.external_provider_payloads.clone()).unwrap();
    assert_eq!(
        providers.canonical_sha256(),
        record.external_provider_archive_sha256()
    );
    assert_eq!(
        providers.canonical_length(),
        record.external_provider_archive_length()
    );
    assert_eq!(
        providers.payload_length(),
        record.external_provider_payload_length()
    );
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&replay.outer_handoff)),
        record.outer_handoff_sha256()
    );
    assert_eq!(replay.outer_handoff.len(), record.outer_handoff_length());
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&replay.transcript)),
        record.transcript_sha256()
    );
    assert_eq!(replay.transcript.len(), record.transcript_length());
    assert_eq!(
        <[u8; 32]>::from(Sha256::digest(&replay.finalized_hsaco)),
        record.output_sha256()
    );
    assert_eq!(replay.finalized_hsaco.len(), record.output_length());
    let current = lease.acquire_current_token().unwrap();
    lease.validate_current_token(&current).unwrap();
    assert_eq!(current.exact_artifact_bytes(), exact_finalized);
    current.revalidate_locked_currentness().unwrap();
    drop(current);
    let outer = InertSemanticCompilerModuleHandoffV3::decode(&replay.outer_handoff).unwrap();
    assert_eq!(
        *outer.capsule().compiler_closure(),
        binding.compiler_closure()
    );
    let transcript =
        ProtectedWorkerV3CompactFinalizerReplayV2::decode_canonical(&replay.transcript).unwrap();
    assert_eq!(
        transcript.expected_finalization_identity(),
        &binding.finalization_identity()
    );
    assert_eq!(
        transcript.source_evidence_identity(),
        &binding.source_evidence_identity()
    );
}

#[test]
fn strict_v3_gfx942_no_ffi_rejects_external_provider_input() {
    let directory = TestDirectory::new();
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    let provider = WorkerInputV1::new(
        WorkerInputKindV1::AmdGpuRelocatable,
        b"unadmitted-external-provider".to_vec(),
    )
    .unwrap();
    let (_, source) = evidence_in_directory_for_kernel_and_providers(
        &directory,
        fixture.bytes,
        EvidenceConfig::BASE,
        "vecadd",
        "vecadd.kd",
        vec![provider],
    );
    assert_eq!(
        inspect_protected_worker_v3_hsaco_v1(source).unwrap_err(),
        WorkerV3HsacoInspectionError::StrictV3Gfx942OcmlProviderClosureMismatch
    );
}

#[test]
fn native_v3_publication_rejects_a_different_producer() {
    let directory = TestDirectory::new();
    let fixture = slice_fixture_with_descriptor_table(&slice_descriptor_table());
    let (_, source) = evidence_in_directory_for_kernel(
        &directory,
        fixture.bytes,
        EvidenceConfig::BASE,
        "vecadd",
        "vecadd.kd",
    );
    let inspected = inspect_protected_worker_v3_hsaco_v1(source).unwrap();
    let finalized = finalize_protected_worker_v3_hsaco_v1(inspected).unwrap();
    let prepared =
        prepare_protected_worker_v3_hsaco_publication_v1(&producer(), finalized).unwrap();
    let other = ProducerIdentity::from_codegen(
        "worker_v3_hsaco_admission_other",
        Some(Path::new("tests/worker_v3_hsaco_admission_other.rs")),
    )
    .unwrap();

    assert!(matches!(
        persist_prepared_protected_worker_v3_hsaco_publication_v1(&directory.0, &other, prepared,),
        Err(fe2o3_hsaco_finalize::WorkerV3HsacoPublicationErrorV1::ProducerIdentityMismatch)
    ));
}

#[test]
fn invocation_closure_transaction_plan_and_worker_axes_cannot_be_dropped() {
    let fixture = || scalar_add_fixture_with(ScalarAddFixtureMutation::RequiredWorkgroup).bytes;
    let base = inspected(fixture(), EvidenceConfig::BASE);
    let changed_attempt = inspected(
        fixture(),
        EvidenceConfig {
            attempt_seed: 0x62,
            ..EvidenceConfig::BASE
        },
    );
    let changed_invocation = inspected(
        fixture(),
        EvidenceConfig {
            invocation_seed: 0x40,
            ..EvidenceConfig::BASE
        },
    );
    let changed_module = inspected(
        fixture(),
        EvidenceConfig {
            module_seed: 0x12,
            ..EvidenceConfig::BASE
        },
    );
    let changed_plan = inspected(
        fixture(),
        EvidenceConfig {
            optimization: "3",
            ..EvidenceConfig::BASE
        },
    );
    let changed_worker = inspected(
        fixture(),
        EvidenceConfig {
            llvm_build_identity: "upstream-llvm-test-build-b",
            ..EvidenceConfig::BASE
        },
    );

    for changed in [
        &changed_attempt,
        &changed_invocation,
        &changed_module,
        &changed_plan,
        &changed_worker,
    ] {
        assert_eq!(base.exact_bytes(), changed.exact_bytes());
        assert_eq!(base.raw_hsaco_identity(), changed.raw_hsaco_identity());
        assert_ne!(base.identity(), changed.identity());
    }
    assert_ne!(base.attempt(), changed_attempt.attempt());
    assert_ne!(
        base.transaction_identity(),
        changed_attempt.transaction_identity()
    );
    assert_ne!(
        base.binding_expectation().invocation_digest(),
        changed_invocation.binding_expectation().invocation_digest()
    );
    assert_ne!(
        base.compiler_closure(),
        changed_invocation.compiler_closure()
    );
    assert_ne!(
        base.outer_handoff_identity(),
        changed_module.outer_handoff_identity()
    );
    assert_ne!(
        base.link_plan_identity(),
        changed_module.link_plan_identity()
    );
    assert_ne!(base.link_plan_identity(), changed_plan.link_plan_identity());
    assert_ne!(
        base.worker_measurement().llvm_build_identity(),
        changed_worker.worker_measurement().llvm_build_identity()
    );
}

#[test]
fn raw_bytes_and_every_structural_hsaco_axis_are_checked() {
    let valid = inspected(
        scalar_add_fixture_with(ScalarAddFixtureMutation::RequiredWorkgroup).bytes,
        EvidenceConfig::BASE,
    );
    let mut changed_fixture = scalar_add_fixture_with(ScalarAddFixtureMutation::RequiredWorkgroup);
    changed_fixture.bytes[changed_fixture.text_offset] ^= 1;
    let changed_bytes = inspected(
        changed_fixture.bytes,
        EvidenceConfig {
            attempt_seed: 0x63,
            ..EvidenceConfig::BASE
        },
    );
    assert_ne!(valid.exact_bytes(), changed_bytes.exact_bytes());
    assert_ne!(
        valid.raw_hsaco_identity(),
        changed_bytes.raw_hsaco_identity()
    );
    assert_ne!(valid.identity(), changed_bytes.identity());

    for (attempt_seed, mutation) in [
        (0x70, ScalarAddFixtureMutation::Target),
        (0x71, ScalarAddFixtureMutation::CodeObjectVersion),
        (0x72, ScalarAddFixtureMutation::EntrySymbol),
        (0x74, ScalarAddFixtureMutation::None),
        (0x77, ScalarAddFixtureMutation::DescriptorComputePgmRsrc1),
        (0x78, ScalarAddFixtureMutation::TruncatedHeader),
    ] {
        let evidence = evidence(
            scalar_add_fixture_with(mutation).bytes,
            EvidenceConfig {
                attempt_seed,
                ..EvidenceConfig::BASE
            },
        );
        assert!(inspect_protected_worker_v3_hsaco_v1(evidence).is_err());
    }
}

fn inspected(bytes: Vec<u8>, config: EvidenceConfig) -> InspectedProtectedWorkerV3HsacoV1 {
    inspect_protected_worker_v3_hsaco_v1(evidence(bytes, config)).unwrap()
}

fn require_v3_inspection(_: &InspectedProtectedWorkerV3HsacoV1) {}

fn evidence(hsaco: Vec<u8>, config: EvidenceConfig) -> InertProtectedFirstBuildWorkerV3EvidenceV1 {
    let directory = TestDirectory::new();
    evidence_in_directory(&directory, hsaco, config).1
}

fn evidence_in_directory(
    directory: &TestDirectory,
    hsaco: Vec<u8>,
    config: EvidenceConfig,
) -> (
    fe2o3_artifact_transaction::BuildAttempt,
    InertProtectedFirstBuildWorkerV3EvidenceV1,
) {
    evidence_in_directory_for_kernel(directory, hsaco, config, "scalar_add", "scalar_add.kd")
}

fn evidence_in_directory_for_kernel(
    directory: &TestDirectory,
    hsaco: Vec<u8>,
    config: EvidenceConfig,
    entry_symbol: &str,
    descriptor_symbol: &str,
) -> (
    fe2o3_artifact_transaction::BuildAttempt,
    InertProtectedFirstBuildWorkerV3EvidenceV1,
) {
    evidence_in_directory_for_kernel_and_providers(
        directory,
        hsaco,
        config,
        entry_symbol,
        descriptor_symbol,
        Vec::new(),
    )
}

fn evidence_in_directory_for_kernel_and_providers(
    directory: &TestDirectory,
    hsaco: Vec<u8>,
    config: EvidenceConfig,
    entry_symbol: &str,
    descriptor_symbol: &str,
    external_providers: Vec<WorkerInputV1>,
) -> (
    fe2o3_artifact_transaction::BuildAttempt,
    InertProtectedFirstBuildWorkerV3EvidenceV1,
) {
    evidence_in_directory_for_kernels_and_providers(
        directory,
        hsaco,
        config,
        &[(entry_symbol, descriptor_symbol)],
        external_providers,
    )
}

fn evidence_in_directory_for_kernels_and_providers(
    directory: &TestDirectory,
    hsaco: Vec<u8>,
    config: EvidenceConfig,
    kernel_symbols: &[(&str, &str)],
    external_providers: Vec<WorkerInputV1>,
) -> (
    fe2o3_artifact_transaction::BuildAttempt,
    InertProtectedFirstBuildWorkerV3EvidenceV1,
) {
    let attempt = begin_build_attempt(
        &directory.0,
        &producer(),
        BuildInvocation::from_bytes([config.attempt_seed; 32]),
        BuildSession::from_bytes([config.attempt_seed.wrapping_add(1); 16]),
    )
    .unwrap();
    let handoff = outer_for_kernels(
        config.invocation_seed,
        config.module_seed,
        &hsaco,
        kernel_symbols,
        config.lineage_mutation,
    );
    let receipt = publish_compiler_module_handoff_in_slot_v3(
        &directory.0,
        &producer(),
        attempt,
        config.slot,
        &handoff,
    )
    .unwrap();
    let consumed = consume_compiler_module_handoff_in_slot_v3(
        &directory.0,
        &producer(),
        attempt,
        config.slot,
        handoff.identity(),
    )
    .unwrap();
    let worker = pinned(directory, config.llvm_build_identity);
    let evidence = execute(config, receipt, consumed, &worker, external_providers);
    (attempt, evidence)
}

fn execute(
    config: EvidenceConfig,
    receipt: CompilerModuleHandoffReceiptV3,
    consumed: ConsumedCompilerModuleHandoffV3,
    worker: &PinnedWorkerV1,
    external_providers: Vec<WorkerInputV1>,
) -> InertProtectedFirstBuildWorkerV3EvidenceV1 {
    let closure = *consumed.handoff().capsule().compiler_closure();
    execute_protected_reproducible_first_build_worker_v3(
        consumed,
        receipt,
        closure,
        worker,
        external_providers,
        options(config.optimization),
        WorkerOutputConstraintsV1::new(1024 * 1024).unwrap(),
        WorkerExecutionLimitsV1::new(Duration::from_secs(3), 2 * 1024 * 1024, 64 * 1024).unwrap(),
    )
    .unwrap()
}

fn target() -> DeviceTargetV1 {
    DeviceTargetV1::parse(TARGET).unwrap()
}

fn slice_descriptor_table() -> Vec<u8> {
    slice_descriptor_table_with_workgroup(64)
}

fn slice_descriptor_table_with_workgroup(workgroup_size: u32) -> Vec<u8> {
    let source = SourceTypeRecordV1::new(SourceTypeDescriptorV1::shared_slice(ScalarTypeV1::F32));
    let layout =
        DeviceLayoutRecordV1::new(DeviceLayoutDescriptorV1::shared_slice(ScalarTypeV1::F32));
    let kernel = slice_kernel_descriptor(
        0xa1,
        "vecadd",
        "vecadd",
        "vecadd.kd",
        &source,
        &layout,
        workgroup_size,
    );
    let table = DeviceDescriptorTableV1::new(
        CanonicalCodeObjectDigest::from_bytes([0; 32]),
        DescriptorCodeObjectVersion::V6,
        CompilerIdentityV1::new(
            Text::new("rustc-codegen-fe2o3").unwrap(),
            Text::new("test").unwrap(),
            [0xa6; 20],
        ),
        ProducerIdentityV1::new(
            Text::new("rustc-codegen-fe2o3-worker-v3").unwrap(),
            Text::new("test").unwrap(),
        ),
        target(),
        vec![source],
        vec![layout],
        vec![kernel],
    )
    .unwrap();
    encode_device_descriptor_table_v1(&table).unwrap()
}

fn synthetic_two_kernel_slice_descriptor_table() -> Vec<u8> {
    let source = SourceTypeRecordV1::new(SourceTypeDescriptorV1::shared_slice(ScalarTypeV1::F32));
    let layout =
        DeviceLayoutRecordV1::new(DeviceLayoutDescriptorV1::shared_slice(ScalarTypeV1::F32));
    let kernels = vec![
        slice_kernel_descriptor(
            0xc1,
            "synthetic_first_transform",
            "synthetic_first_transform",
            "synthetic_first_transform.kd",
            &source,
            &layout,
            64,
        ),
        slice_kernel_descriptor(
            0xb1,
            "synthetic_second_transform",
            "synthetic_second_transform",
            "synthetic_second_transform.kd",
            &source,
            &layout,
            64,
        ),
    ];
    let table = DeviceDescriptorTableV1::new(
        CanonicalCodeObjectDigest::from_bytes([0; 32]),
        DescriptorCodeObjectVersion::V6,
        CompilerIdentityV1::new(
            Text::new("rustc-codegen-fe2o3").unwrap(),
            Text::new("test").unwrap(),
            [0xa6; 20],
        ),
        ProducerIdentityV1::new(
            Text::new("rustc-codegen-fe2o3-worker-v3").unwrap(),
            Text::new("test").unwrap(),
        ),
        target(),
        vec![source],
        vec![layout],
        kernels,
    )
    .unwrap();
    encode_device_descriptor_table_v1(&table).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn slice_kernel_descriptor(
    identity_seed: u8,
    logical_name: &str,
    entry_name: &str,
    descriptor_symbol: &str,
    source: &SourceTypeRecordV1,
    layout: &DeviceLayoutRecordV1,
    workgroup_size: u32,
) -> KernelDescriptorV1 {
    KernelDescriptorV1::new(
        KernelId::from_bytes([identity_seed; 32]),
        ValidName::new(logical_name).unwrap(),
        ValidName::new(entry_name).unwrap(),
        ValidName::new(descriptor_symbol).unwrap(),
        BuildEvidenceV1::new(
            EvidenceIdentity::from_opaque_bytes([identity_seed.wrapping_add(1); 32]),
            EvidenceDigest::from_sha256_bytes([identity_seed.wrapping_add(2); 32]),
        ),
        BuildEvidenceV1::new(
            EvidenceIdentity::from_opaque_bytes([identity_seed.wrapping_add(3); 32]),
            EvidenceDigest::from_sha256_bytes([identity_seed.wrapping_add(4); 32]),
        ),
        Vec::new(),
        KernelAbiLayoutV1::new(16, 272, 8).unwrap(),
        LaunchConstraintsV1::new(
            1,
            BlockSizeV1::Exact(DimensionsV1::new(workgroup_size, 1, 1).unwrap()),
            DimensionsV1::new(u32::MAX, 1, 1).unwrap(),
            workgroup_size,
            0,
            64 * 1024,
        )
        .unwrap(),
        vec![
            LogicalArgumentV1::shared_slice(
                0,
                ValidName::new("values").unwrap(),
                source,
                layout,
                0,
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn worker_path() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_fe2o3-worker-v3-hsaco-fixture"))
}

fn pinned(directory: &TestDirectory, llvm_build_identity: &str) -> PinnedWorkerV1 {
    let private_worker = directory.0.join("worker");
    fs::copy(worker_path(), &private_worker).unwrap();
    let bytes = fs::read(&private_worker).unwrap();
    let measurement = WorkerMeasurementV1::new(
        ContentIdentityV1::calculate(&bytes),
        WORKER_BUILD_ID,
        llvm_build_identity,
    )
    .unwrap();
    PinnedWorkerV1::open(private_worker, measurement).unwrap()
}

fn options(optimization: &str) -> Vec<LinkOptionV1> {
    [
        ("verify-each", "true"),
        ("code-object-version", "6"),
        ("strip-debug", "true"),
        ("opt-level", optimization),
    ]
    .into_iter()
    .map(|(name, value)| LinkOptionV1::new(name, value).unwrap())
    .collect()
}

fn module_handoff_for_kernels(
    seed: u8,
    hsaco: &[u8],
    kernel_symbols: &[(&str, &str)],
) -> CompilerModuleHandoffV2 {
    let mut module = format!("; ModuleID = 'raw-hsaco-v3-{seed:02x}'\n").into_bytes();
    module.extend_from_slice(RAW_HSACO_MARKER);
    module.extend_from_slice(hex_encode(hsaco).as_bytes());
    module.push(b'\n');
    let envelope =
        CompilerFfiEnvelopeV1::for_module_without_device_ffi(target(), CodeObjectVersion::V6)
            .unwrap();
    let mut symbols = kernel_symbols
        .iter()
        .flat_map(|(entry_symbol, descriptor_symbol)| {
            [
                (CompilerModuleSymbolRoleV1::KernelEntry, *entry_symbol),
                (
                    CompilerModuleSymbolRoleV1::KernelDescriptor,
                    *descriptor_symbol,
                ),
            ]
        })
        .collect::<Vec<_>>();
    symbols.sort_unstable();
    let manifest = CompilerModuleSymbolManifestV1::new(symbols).unwrap();
    CompilerModuleHandoffV2::new(
        CompilerModuleKindV1::LlvmTextIr,
        target(),
        CodeObjectVersion::V6,
        envelope,
        manifest,
        &module,
    )
    .unwrap()
}

fn outer_for_kernels(
    invocation_seed: u8,
    module_seed: u8,
    hsaco: &[u8],
    kernel_symbols: &[(&str, &str)],
    lineage_mutation: DescriptorLineageMutation,
) -> InertSemanticCompilerModuleHandoffV3 {
    let handoff = module_handoff_for_kernels(module_seed, hsaco, kernel_symbols);
    InertSemanticCompilerModuleHandoffV3::decode(&raw_outer(
        &capsule_bytes(invocation_seed, &handoff, lineage_mutation),
        handoff.canonical_bytes(),
    ))
    .unwrap()
}

fn capsule_bytes(
    seed: u8,
    handoff: &CompilerModuleHandoffV2,
    lineage_mutation: DescriptorLineageMutation,
) -> Vec<u8> {
    let invocation = invocation_bytes(seed);
    let final_commitment = InertFinalCompilerModuleCommitmentV3::from_handoff(handoff).unwrap();
    let mut receipts = RECEIPTS
        .iter()
        .map(|(label, domain)| {
            (
                format!("worker-v3/receipt/{label}/{seed:02x}").into_bytes(),
                *domain,
            )
        })
        .collect::<Vec<_>>();
    let proof_inputs = canonical_compiler_proof_inputs_v4(seed);
    receipts[2].0 = proof_inputs.semantic_mir().to_vec();
    receipts[3].0 = proof_inputs.middle_end().to_vec();
    receipts[4].0 = proof_inputs.kernel_ir().to_vec();
    receipts[5].0 = proof_inputs.correspondence().to_vec();
    receipts[6].0 = proof_inputs.formal_memory().to_vec();
    let hsaco = handoff
        .module_bytes()
        .windows(RAW_HSACO_MARKER.len())
        .position(|window| window == RAW_HSACO_MARKER)
        .and_then(|offset| {
            let encoded = &handoff.module_bytes()[offset + RAW_HSACO_MARKER.len()..];
            let line = encoded.split(|byte| *byte == b'\n').next()?;
            hex_decode(line)
        });
    if let Some(descriptor_source) = hsaco
        .as_deref()
        .and_then(|bytes| inspect_unfinalized(bytes).ok())
        .and_then(|inspection| {
            encode_device_descriptor_table_v1(inspection.descriptor_table()).ok()
        })
    {
        receipts[10].0 = descriptor_source;
    }
    let verus_execution = canonical_verus_execution_evidence_v1(&receipts[3].0, seed);
    receipts[7].0 = proof_binding_association_payload(&receipts, &verus_execution);
    receipts[11].0 = handoff.symbol_manifest().canonical_bytes().to_vec();
    match lineage_mutation {
        DescriptorLineageMutation::Exact => {}
        DescriptorLineageMutation::DifferentCanonicalSource => {
            let source = &mut receipts[10].0;
            let offset = source
                .windows(4)
                .position(|window| window == b"test")
                .expect("fixture descriptor has a test identity");
            source[offset] = b'b';
            CompilerDescriptorSourceV1::decode(source)
                .expect("hostile source remains canonical and zero-digest");
        }
        DescriptorLineageMutation::DifferentExportManifest => {
            receipts[11].0 = b"different canonical export manifest receipt".to_vec();
        }
    }
    receipts.push((
        final_commitment.canonical_bytes().to_vec(),
        FINAL_RECEIPT_DOMAIN_V3,
    ));
    let total_len = 24
        + 4
        + invocation.len()
        + 32
        + 2
        + TARGET.len()
        + receipts
            .iter()
            .map(|(payload, _)| 4 + payload.len() + 32)
            .sum::<usize>()
        + 32;
    let mut capsule = Vec::with_capacity(total_len);
    capsule.extend_from_slice(CAPSULE_MAGIC_V3);
    capsule.extend_from_slice(&CAPSULE_VERSION_V3.to_le_bytes());
    capsule.extend_from_slice(&0_u16.to_le_bytes());
    capsule.extend_from_slice(&(total_len as u64).to_le_bytes());
    capsule.extend_from_slice(&0_u32.to_le_bytes());
    push_blob(&mut capsule, &invocation);
    capsule.extend_from_slice(&identity(INVOCATION_DIGEST_DOMAIN_V3, &invocation));
    capsule.extend_from_slice(&(TARGET.len() as u16).to_le_bytes());
    capsule.extend_from_slice(TARGET.as_bytes());
    for (payload, domain) in receipts {
        push_blob(&mut capsule, &payload);
        capsule.extend_from_slice(&identity(domain, &payload));
    }
    let capsule_identity = identity(CAPSULE_IDENTITY_DOMAIN_V3, &capsule);
    capsule.extend_from_slice(&capsule_identity);
    assert_eq!(capsule.len(), total_len);
    capsule
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn hex_decode(encoded: &[u8]) -> Option<Vec<u8>> {
    if encoded.len() % 2 != 0 {
        return None;
    }
    encoded
        .chunks_exact(2)
        .map(|pair| Some((decode_hex_nibble(pair[0])? << 4) | decode_hex_nibble(pair[1])?))
        .collect()
}

const fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn proof_binding_association_payload(
    receipts: &[(Vec<u8>, &[u8])],
    verus_execution: &[u8],
) -> Vec<u8> {
    let mut identities = Vec::with_capacity(5);
    for (payload, domain) in receipts.iter().take(7).skip(2) {
        identities.push(
            InertLineageContentIdentityV3::new(identity(domain, payload), payload.len() as u64)
                .unwrap(),
        );
    }
    InertProofBindingAssociationV4::new(
        InertProofBindingAssociationInputsV4::new(
            identities[0],
            identities[1],
            identities[2],
            identities[3],
            identities[4],
        ),
        verus_execution,
    )
    .unwrap()
    .canonical_bytes()
    .to_vec()
}

fn raw_outer(capsule: &[u8], handoff: &[u8]) -> Vec<u8> {
    let capsule_sha256: [u8; 32] = capsule[capsule.len() - 32..].try_into().unwrap();
    let handoff_sha256: [u8; 32] = Sha256::digest(handoff).into();
    let mut pair = Vec::with_capacity(INERT_COMPILER_MODULE_PAIR_BINDING_BYTES_V3);
    pair.extend_from_slice(&INERT_COMPILER_MODULE_PAIR_BINDING_MAGIC_V3);
    pair.extend_from_slice(&INERT_COMPILER_MODULE_PAIR_BINDING_VERSION_V3.to_le_bytes());
    pair.extend_from_slice(&0_u16.to_le_bytes());
    pair.extend_from_slice(&(INERT_COMPILER_MODULE_PAIR_BINDING_BYTES_V3 as u32).to_le_bytes());
    pair.extend_from_slice(&0_u32.to_le_bytes());
    pair.extend_from_slice(&capsule_sha256);
    pair.extend_from_slice(&(capsule.len() as u64).to_le_bytes());
    pair.extend_from_slice(&handoff_sha256);
    pair.extend_from_slice(&(handoff.len() as u64).to_le_bytes());
    let pair_identity = identity(PAIR_IDENTITY_DOMAIN_V3, &pair);
    pair.extend_from_slice(&pair_identity);

    let total_len = 40 + capsule.len() + handoff.len() + pair.len() + 32;
    let mut outer = Vec::with_capacity(total_len);
    outer.extend_from_slice(&INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_MAGIC_V3);
    outer.extend_from_slice(&INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_VERSION_V3.to_le_bytes());
    outer.extend_from_slice(&0_u16.to_le_bytes());
    outer.extend_from_slice(&(total_len as u64).to_le_bytes());
    outer.extend_from_slice(&0_u32.to_le_bytes());
    outer.extend_from_slice(&(capsule.len() as u64).to_le_bytes());
    outer.extend_from_slice(&(handoff.len() as u64).to_le_bytes());
    outer.extend_from_slice(capsule);
    outer.extend_from_slice(handoff);
    outer.extend_from_slice(&pair);
    let outer_identity = identity(OUTER_IDENTITY_DOMAIN_V3, &outer);
    outer.extend_from_slice(&outer_identity);
    assert_eq!(outer.len(), total_len);
    outer
}

fn invocation_bytes(seed: u8) -> Vec<u8> {
    let encoded = match seed {
        0x20 => INVOCATION_20_HEX,
        0x40 => INVOCATION_40_HEX,
        _ => panic!("unsupported strict invocation fixture seed {seed:#x}"),
    };
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]))
        .collect()
}

fn hex_nibble(value: u8) -> u8 {
    match value {
        b'0'..=b'9' => value - b'0',
        b'a'..=b'f' => value - b'a' + 10,
        _ => panic!("non-canonical fixture hex"),
    }
}

fn push_blob(output: &mut Vec<u8>, bytes: &[u8]) {
    output.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(bytes);
}

fn identity(domain: &[u8], preimage: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update((preimage.len() as u64).to_le_bytes());
    digest.update(preimage);
    digest.finalize().into()
}

fn producer() -> ProducerIdentity {
    ProducerIdentity::from_codegen(
        "worker_v3_hsaco_admission_fixture",
        Some(Path::new("tests/worker_v3_hsaco_admission.rs")),
    )
    .unwrap()
}

pub(crate) struct TestDirectory(pub(crate) PathBuf);

impl TestDirectory {
    pub(crate) fn new() -> Self {
        fe2o3_artifact_transaction::enable_same_mount_namespace_artifact_path_guard_v1();
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "fe2o3-worker-v3-admission-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
