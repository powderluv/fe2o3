//! Allocation qualification for the V3 owned decoder.
//!
//! The test materializes the 64 MiB maximum module payload and one 4 MiB
//! maximum non-MIR receipt. Materializing the approximately 240 MiB aggregate
//! outer maximum in every test run would add substantial peak memory while
//! exercising the same range-retention code. Exact formula assertions cover
//! that aggregate boundary, and equal allocation traces across representative
//! and maximum-sized payloads establish that decode allocation topology does
//! not scale with capsule, receipt, nested-handoff, or module payload length.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    ffi::OsString,
    sync::atomic::{AtomicUsize, Ordering},
};

use fe2o3_build_authority::CompilerClosureV2;
use fe2o3_compiler_ffi::{
    CodeObjectVersion, CompilerFfiEnvelopeV1, CompilerModuleHandoffV2, CompilerModuleKindV1,
    CompilerModuleSymbolManifestV1, CompilerModuleSymbolRoleV1, DeviceTargetV1,
    INERT_COMPILER_MODULE_PAIR_BINDING_BYTES_V3, InertFinalCompilerModuleCommitmentV3,
    InertSemanticCompilerModuleHandoffErrorV3, InertSemanticCompilerModuleHandoffV3,
    MAX_COMPILER_FFI_ENVELOPE_BYTES_V1, MAX_COMPILER_MODULE_BYTES_V1,
    MAX_COMPILER_MODULE_HANDOFF_BYTES_V2, MAX_COMPILER_MODULE_SYMBOL_MANIFEST_BYTES_V1,
    MAX_INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_BYTES_V3,
};
use fe2o3_compiler_lineage::{
    InertAbiReceiptV3, InertAmdgpuLoweringReceiptV3, InertCanonicalSemanticMirReceiptV3,
    InertDataLayoutReceiptV3, InertExportManifestReceiptV3,
    InertFinalCompilerModuleCommitmentReceiptV3, InertFormalMemoryReceiptV3,
    InertKernelIrReceiptV3, InertMiddleEndReceiptV3, InertMirToKirCorrespondenceReceiptV3,
    InertProductionSemanticCapsuleV3, InertProofBindingReceiptV3,
    InertRustcIdentityInventoryReceiptV3, InertRustcPreflightPlanReceiptV3,
    InertSemanticToLlvmReceiptV3, InertTargetBindingReceiptV3, MAX_CANONICAL_SEMANTIC_MIR_BYTES_V3,
    MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3, MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3,
    OrderedInertSemanticLineageReceiptsV3,
};
use fe2o3_rustc_invocation::{
    CompileEnvironmentV2, MAX_DESCRIPTOR_BYTES_V3, RustcInvocationDescriptorV2,
    RustcInvocationDescriptorV3, RustcUnitV2,
};
use reserved_fe2o3_symbols::MAX_DEVICE_FFI_TARGET_BYTES_V1;

const TARGET: &str = "gfx942:sramecc+:xnack-";
const REPRESENTATIVE_MODULE_BYTES: usize = 2 * 1024 * 1024;
const REPRESENTATIVE_RECEIPT_BYTES: usize = 256 * 1024;
const MAX_ALLOCATION_EVENTS: usize = 4_096;

const OUTER_HEADER_BYTES_V3: usize = 8 + 2 + 2 + 8 + 4 + 8 + 8;
const OUTER_IDENTITY_BYTES_V3: usize = 32;
const V2_HANDOFF_DOMAIN_BYTES: usize = b"FE2O3/COMPILER-MODULE-HANDOFF/V2\0".len();
const V2_CONTENT_IDENTITY_BYTES: usize = 32 + 8;
const V2_HANDOFF_FIXED_BYTES: usize =
    V2_HANDOFF_DOMAIN_BYTES + 4 + 1 + 1 + V2_CONTENT_IDENTITY_BYTES + 4 + V2_CONTENT_IDENTITY_BYTES;

struct ForwardingCountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: ForwardingCountingAllocator = ForwardingCountingAllocator;

static ALLOCATION_EVENT_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_EVENT_SIZES: [AtomicUsize; MAX_ALLOCATION_EVENTS] =
    [const { AtomicUsize::new(0) }; MAX_ALLOCATION_EVENTS];

thread_local! {
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for ForwardingCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: This allocator forwards the unchanged layout to System.
        let pointer = unsafe { System.alloc(layout) };
        record_successful_allocation(pointer, layout.size());
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: This allocator forwards the unchanged layout to System.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        record_successful_allocation(pointer, layout.size());
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: The pointer and original layout came from System, and the
        // requested new size is forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        record_successful_allocation(new_pointer, new_size);
        new_pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout came from this allocator's System
        // forwarding methods.
        unsafe { System.dealloc(pointer, layout) };
    }
}

fn record_successful_allocation(pointer: *mut u8, size: usize) {
    if pointer.is_null() {
        return;
    }
    let counting = COUNT_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
    if !counting {
        return;
    }
    let index = ALLOCATION_EVENT_COUNT.fetch_add(1, Ordering::Relaxed);
    if let Some(slot) = ALLOCATION_EVENT_SIZES.get(index) {
        slot.store(size, Ordering::Relaxed);
    }
}

#[derive(Debug, Eq, PartialEq)]
struct AllocationSnapshot {
    sizes: Vec<usize>,
    unrecorded_events: usize,
}

struct DisableAllocationCounting;

impl Drop for DisableAllocationCounting {
    fn drop(&mut self) {
        COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
    }
}

fn measure_allocations<T>(operation: impl FnOnce() -> T) -> (T, AllocationSnapshot) {
    COUNT_ALLOCATIONS.with(|enabled| {
        assert!(!enabled.replace(true), "allocation scopes must not nest");
    });
    ALLOCATION_EVENT_COUNT.store(0, Ordering::Relaxed);
    let disable = DisableAllocationCounting;
    let result = operation();
    drop(disable);

    let event_count = ALLOCATION_EVENT_COUNT.load(Ordering::Relaxed);
    let recorded_count = event_count.min(MAX_ALLOCATION_EVENTS);
    let sizes = ALLOCATION_EVENT_SIZES[..recorded_count]
        .iter()
        .map(|size| size.load(Ordering::Relaxed))
        .collect();
    (
        result,
        AllocationSnapshot {
            sizes,
            unrecorded_events: event_count - recorded_count,
        },
    )
}

fn target() -> DeviceTargetV1 {
    DeviceTargetV1::parse(TARGET).expect("canonical gfx942 target")
}

fn invocation() -> RustcInvocationDescriptorV3 {
    let pins = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32], [6; 32]];
    let closure = CompilerClosureV2::new(pins[0], pins[1], pins[2], pins[3], pins[4], pins[5])
        .expect("nonzero compiler closure");
    let rustc = RustcUnitV2::new(
        "/workspace/fe2o3",
        vec![
            "/opt/fe2o3/rustc".into(),
            "--crate-name".into(),
            "v3_owned_decode_allocations".into(),
            "crates/v3-owned-decode-allocations/src/lib.rs".into(),
            "--crate-type=lib".into(),
            "--edition=2024".into(),
            "-Zcodegen-backend=/opt/fe2o3/librustc_codegen_fe2o3.so".into(),
        ],
    )
    .expect("valid rustc fixture");
    let environment = CompileEnvironmentV2::from_child_environment([
        (
            OsString::from("CARGO_CFG_TARGET_ARCH"),
            OsString::from("amdgcn"),
        ),
        (OsString::from("FE2O3_TARGET"), OsString::from(TARGET)),
        (
            OsString::from("FE2O3_HSACO_DIR"),
            OsString::from("/workspace/fe2o3/target/fe2o3"),
        ),
        (
            OsString::from("FE2O3_VERIFY_KERNEL_IR"),
            OsString::from("1"),
        ),
    ])
    .expect("valid exact environment");
    let v2 = RustcInvocationDescriptorV2::new(pins[3], pins[5], rustc, environment)
        .expect("valid V2 invocation");
    RustcInvocationDescriptorV3::new(v2, closure).expect("matching V3 compiler closure")
}

fn envelope() -> CompilerFfiEnvelopeV1 {
    CompilerFfiEnvelopeV1::for_module_without_device_ffi(target(), CodeObjectVersion::V5)
        .expect("valid FFI-free envelope")
}

fn manifest() -> CompilerModuleSymbolManifestV1 {
    CompilerModuleSymbolManifestV1::new([
        (CompilerModuleSymbolRoleV1::KernelEntry, "kernel"),
        (CompilerModuleSymbolRoleV1::KernelDescriptor, "kernel.kd"),
    ])
    .expect("valid symbol manifest")
}

fn module_handoff(module_bytes: &[u8]) -> CompilerModuleHandoffV2 {
    CompilerModuleHandoffV2::new(
        CompilerModuleKindV1::LlvmBitcode,
        target(),
        CodeObjectVersion::V5,
        envelope(),
        manifest(),
        module_bytes,
    )
    .expect("valid V2 module handoff")
}

fn payload(byte_len: usize, byte: u8) -> Vec<u8> {
    vec![byte; byte_len]
}

fn receipts(
    module_handoff: &CompilerModuleHandoffV2,
    first_receipt_bytes: usize,
) -> OrderedInertSemanticLineageReceiptsV3 {
    let final_commitment = InertFinalCompilerModuleCommitmentV3::from_handoff(module_handoff)
        .expect("valid final module commitment");
    OrderedInertSemanticLineageReceiptsV3::new(
        InertRustcIdentityInventoryReceiptV3::from_canonical_preimage(payload(
            first_receipt_bytes,
            1,
        ))
        .unwrap(),
        InertRustcPreflightPlanReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            2,
        ))
        .unwrap(),
        InertCanonicalSemanticMirReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            3,
        ))
        .unwrap(),
        InertMiddleEndReceiptV3::from_canonical_preimage(payload(REPRESENTATIVE_RECEIPT_BYTES, 4))
            .unwrap(),
        InertKernelIrReceiptV3::from_canonical_preimage(payload(REPRESENTATIVE_RECEIPT_BYTES, 5))
            .unwrap(),
        InertMirToKirCorrespondenceReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            6,
        ))
        .unwrap(),
        InertFormalMemoryReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            7,
        ))
        .unwrap(),
        InertProofBindingReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            8,
        ))
        .unwrap(),
        InertTargetBindingReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            9,
        ))
        .unwrap(),
        InertDataLayoutReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            10,
        ))
        .unwrap(),
        InertAbiReceiptV3::from_canonical_preimage(payload(REPRESENTATIVE_RECEIPT_BYTES, 11))
            .unwrap(),
        InertExportManifestReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            12,
        ))
        .unwrap(),
        InertAmdgpuLoweringReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            13,
        ))
        .unwrap(),
        InertSemanticToLlvmReceiptV3::from_canonical_preimage(payload(
            REPRESENTATIVE_RECEIPT_BYTES,
            14,
        ))
        .unwrap(),
        InertFinalCompilerModuleCommitmentReceiptV3::from_canonical_preimage(
            final_commitment.canonical_bytes().to_vec(),
        )
        .unwrap(),
    )
}

fn owned_outer_wire(module_bytes: usize, first_receipt_bytes: usize) -> Box<[u8]> {
    let module = payload(module_bytes, 0x42);
    let handoff = module_handoff(&module);
    drop(module);
    let capsule = InertProductionSemanticCapsuleV3::new(
        invocation(),
        target(),
        receipts(&handoff, first_receipt_bytes),
    )
    .expect("valid V3 capsule");
    let outer = InertSemanticCompilerModuleHandoffV3::new(capsule, handoff)
        .expect("valid V3 outer handoff");
    outer.canonical_bytes().to_vec().into_boxed_slice()
}

fn receipt_preimages(capsule: &InertProductionSemanticCapsuleV3) -> [&[u8]; 15] {
    let receipts = capsule.receipts();
    [
        receipts.rustc_identity_inventory().canonical_preimage(),
        receipts.rustc_preflight_plan().canonical_preimage(),
        receipts.semantic_mir().canonical_preimage(),
        receipts.middle_end().canonical_preimage(),
        receipts.kernel_ir().canonical_preimage(),
        receipts.mir_to_kir_correspondence().canonical_preimage(),
        receipts.formal_memory().canonical_preimage(),
        receipts.proof_binding().canonical_preimage(),
        receipts.target_binding().canonical_preimage(),
        receipts.data_layout().canonical_preimage(),
        receipts.abi().canonical_preimage(),
        receipts.export_manifest().canonical_preimage(),
        receipts.amdgpu_lowering().canonical_preimage(),
        receipts.semantic_to_llvm().canonical_preimage(),
        receipts
            .final_compiler_module_commitment()
            .canonical_preimage(),
    ]
}

fn assert_retained_range(outer: &[u8], retained: &[u8], label: &str) {
    let outer_start = outer.as_ptr() as usize;
    let outer_end = outer_start
        .checked_add(outer.len())
        .expect("outer pointer range");
    let retained_start = retained.as_ptr() as usize;
    let retained_end = retained_start
        .checked_add(retained.len())
        .expect("retained pointer range");
    assert!(
        retained_start >= outer_start && retained_end <= outer_end,
        "{label} must retain a range in the exact owned outer allocation"
    );
}

fn assert_single_backing(decoded: &InertSemanticCompilerModuleHandoffV3, owned_pointer: *const u8) {
    let outer = decoded.canonical_bytes();
    assert_eq!(outer.as_ptr(), owned_pointer);
    assert_retained_range(outer, decoded.capsule().canonical_bytes(), "capsule");
    for (index, preimage) in receipt_preimages(decoded.capsule()).into_iter().enumerate() {
        assert_retained_range(outer, preimage, &format!("receipt {index}"));
    }
    assert_retained_range(
        outer,
        decoded.module_handoff().canonical_bytes(),
        "nested V2 handoff",
    );
    assert_retained_range(
        outer,
        decoded.module_handoff().module_bytes(),
        "module payload",
    );
}

fn assert_no_payload_sized_allocations(
    decoded: &InertSemanticCompilerModuleHandoffV3,
    snapshot: &AllocationSnapshot,
) {
    assert_eq!(
        snapshot.unrecorded_events, 0,
        "allocation event buffer filled"
    );
    let outer_len = decoded.canonical_bytes().len();
    let capsule_len = decoded.capsule().canonical_bytes().len();
    let handoff_len = decoded.module_handoff().canonical_bytes().len();
    let module_len = decoded.module_handoff().module_bytes().len();
    let receipt_lengths: Vec<_> = receipt_preimages(decoded.capsule())
        .into_iter()
        .map(<[u8]>::len)
        .filter(|length| *length >= REPRESENTATIVE_RECEIPT_BYTES)
        .collect();

    // Small decoded metadata, including validation-time reconstruction of the
    // compact final-module commitment, remains below this payload floor. The
    // equal-trace assertion across input sizes below proves those allocations
    // are fixed by schema shape rather than retained payload length.
    assert!(
        snapshot
            .sizes
            .iter()
            .all(|size| *size < REPRESENTATIVE_RECEIPT_BYTES),
        "decode requested payload-scale allocation(s): {:?}",
        snapshot.sizes
    );
    for forbidden in [outer_len, capsule_len, handoff_len, module_len] {
        assert!(!snapshot.sizes.contains(&forbidden));
    }
    for forbidden in receipt_lengths {
        assert!(!snapshot.sizes.contains(&forbidden));
    }
}

fn assert_maximum_bound_formulas() {
    let expected_v2_max = V2_HANDOFF_FIXED_BYTES
        .checked_add(MAX_DEVICE_FFI_TARGET_BYTES_V1)
        .and_then(|bytes| bytes.checked_add(MAX_COMPILER_FFI_ENVELOPE_BYTES_V1))
        .and_then(|bytes| bytes.checked_add(MAX_COMPILER_MODULE_SYMBOL_MANIFEST_BYTES_V1))
        .and_then(|bytes| bytes.checked_add(MAX_COMPILER_MODULE_BYTES_V1))
        .unwrap();
    assert_eq!(MAX_COMPILER_MODULE_HANDOFF_BYTES_V2, expected_v2_max);

    let expected_outer_max = OUTER_HEADER_BYTES_V3
        .checked_add(MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3)
        .and_then(|bytes| bytes.checked_add(MAX_COMPILER_MODULE_HANDOFF_BYTES_V2))
        .and_then(|bytes| bytes.checked_add(INERT_COMPILER_MODULE_PAIR_BINDING_BYTES_V3))
        .and_then(|bytes| bytes.checked_add(OUTER_IDENTITY_BYTES_V3))
        .unwrap();
    assert_eq!(
        MAX_INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_BYTES_V3,
        expected_outer_max
    );
    assert_eq!(MAX_COMPILER_MODULE_BYTES_V1, 64 * 1024 * 1024);
    assert_eq!(MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3, 4 * 1024 * 1024);
    assert_eq!(MAX_CANONICAL_SEMANTIC_MIR_BYTES_V3, 128 * 1024 * 1024);
    assert_eq!(
        MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3,
        160 * 1024 * 1024
    );
    assert_eq!(
        InertSemanticCompilerModuleHandoffV3::OWNED_DECODE_ADDITIONAL_OUTER_BUFFERS,
        0
    );
    assert_eq!(
        InertProductionSemanticCapsuleV3::MAX_SUCCESSFUL_DECODE_RETAINED_BYTES,
        MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3 + MAX_DESCRIPTOR_BYTES_V3
    );
}

fn assert_hostile_declared_length_rejected_without_allocation(
    mut wire: Box<[u8]>,
    offset: usize,
    declared: u64,
    expected: InertSemanticCompilerModuleHandoffErrorV3,
) {
    wire[offset..offset + 8].copy_from_slice(&declared.to_le_bytes());
    let (result, snapshot) =
        measure_allocations(|| InertSemanticCompilerModuleHandoffV3::decode_owned(wire));
    assert_eq!(result, Err(expected));
    assert_eq!(snapshot.sizes, [] as [usize; 0]);
    assert_eq!(snapshot.unrecorded_events, 0);
}

#[test]
fn v3_owned_decode_allocation_qualification() {
    assert_native_carrier_has_no_payload_copies();
    assert_native_v4_has_no_payload_copies();
    assert_maximum_bound_formulas();

    let representative =
        owned_outer_wire(REPRESENTATIVE_MODULE_BYTES, REPRESENTATIVE_RECEIPT_BYTES);
    let representative_pointer = representative.as_ptr();
    let hostile_wire = representative.clone();
    let (representative_result, representative_allocations) =
        measure_allocations(|| InertSemanticCompilerModuleHandoffV3::decode_owned(representative));
    let representative_decoded = representative_result.expect("representative owned decode");
    assert_single_backing(&representative_decoded, representative_pointer);
    assert_no_payload_sized_allocations(&representative_decoded, &representative_allocations);
    drop(representative_decoded);

    let maximum_module = owned_outer_wire(
        MAX_COMPILER_MODULE_BYTES_V1,
        MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3,
    );
    let maximum_module_pointer = maximum_module.as_ptr();
    let (maximum_result, maximum_allocations) =
        measure_allocations(|| InertSemanticCompilerModuleHandoffV3::decode_owned(maximum_module));
    let maximum_decoded = maximum_result.expect("maximum-module owned decode");
    assert_eq!(
        maximum_decoded.module_handoff().module_bytes().len(),
        MAX_COMPILER_MODULE_BYTES_V1
    );
    assert_eq!(
        maximum_decoded
            .capsule()
            .receipts()
            .rustc_identity_inventory()
            .canonical_preimage()
            .len(),
        MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3
    );
    assert_single_backing(&maximum_decoded, maximum_module_pointer);
    assert_no_payload_sized_allocations(&maximum_decoded, &maximum_allocations);
    assert_eq!(
        maximum_allocations, representative_allocations,
        "decode allocation topology must be independent of retained payload lengths"
    );
    drop(maximum_decoded);

    const TOTAL_LEN_OFFSET: usize = 8 + 2 + 2;
    const CAPSULE_LEN_OFFSET: usize = TOTAL_LEN_OFFSET + 8 + 4;
    const MODULE_HANDOFF_LEN_OFFSET: usize = CAPSULE_LEN_OFFSET + 8;
    assert_hostile_declared_length_rejected_without_allocation(
        hostile_wire.clone(),
        TOTAL_LEN_OFFSET,
        (MAX_INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_BYTES_V3 as u64) + 1,
        InertSemanticCompilerModuleHandoffErrorV3::OuterByteBoundExceeded,
    );
    assert_hostile_declared_length_rejected_without_allocation(
        hostile_wire.clone(),
        CAPSULE_LEN_OFFSET,
        (MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3 as u64) + 1,
        InertSemanticCompilerModuleHandoffErrorV3::CapsuleByteBoundExceeded,
    );
    assert_hostile_declared_length_rejected_without_allocation(
        hostile_wire,
        MODULE_HANDOFF_LEN_OFFSET,
        (MAX_COMPILER_MODULE_HANDOFF_BYTES_V2 as u64) + 1,
        InertSemanticCompilerModuleHandoffErrorV3::ModuleHandoffByteBoundExceeded,
    );
}

fn native_v4_wire(
    module_bytes: usize,
    receipt_bytes: usize,
    output_bytes: usize,
    source_bytes: usize,
) -> Vec<u8> {
    use fe2o3_compiler_ffi::{
        InertSemanticCompilerModuleHandoffLayoutV4 as HandoffLayout,
        seal_inert_semantic_compiler_module_handoff_v4,
    };
    use fe2o3_compiler_lineage::{
        InertProductionSemanticCapsuleLayoutV4 as CapsuleLayout,
        MAX_NATIVE_REFINED_FORWARDING_CARRIER_STORAGE_V1 as LIMIT,
        NativeRefinedForwardingCarrierLayoutV1 as CarrierLayout,
        seal_inert_production_semantic_capsule_v4, seal_native_refined_forwarding_carrier_v1,
    };
    let old = InertSemanticCompilerModuleHandoffV3::decode_owned(owned_outer_wire(
        module_bytes,
        receipt_bytes,
    ))
    .unwrap();
    let pair = CarrierLayout::new::<()>(output_bytes, source_bytes).unwrap();
    let cap = CapsuleLayout::new::<()>(old.capsule().canonical_bytes().len(), pair.encoded_len())
        .unwrap();
    let outer = HandoffLayout::new(
        cap.encoded_len(),
        old.module_handoff().canonical_bytes().len(),
    )
    .unwrap();
    // All three encoders seal directly in the same final allocation.
    let mut bytes = vec![0x17; outer.encoded_len()];
    let capsule = &mut bytes[outer.capsule_range()];
    capsule[cap.base_range()].copy_from_slice(old.capsule().canonical_bytes());
    let (_, trace) = measure_allocations(|| {
        seal_native_refined_forwarding_carrier_v1(
            pair,
            &mut capsule[cap.carrier_range()],
            LIMIT,
            |_| Ok::<_, ()>(()),
        )
        .unwrap();
    });
    assert_eq!(trace.sizes, [] as [usize; 0]);
    assert_eq!(trace.unrecorded_events, 0);
    let (cap_identity, trace) = measure_allocations(|| {
        let identity =
            seal_inert_production_semantic_capsule_v4(cap, capsule, LIMIT, |_| Ok::<_, ()>(()))
                .unwrap();
        let view = fe2o3_compiler_lineage::read_inert_production_semantic_capsule_v4(
            capsule,
            LIMIT,
            |_| Ok::<_, ()>(()),
        )
        .unwrap();
        assert_eq!(view.identity(), identity);
        identity
    });
    assert_eq!(trace.sizes, [] as [usize; 0]);
    assert_eq!(trace.unrecorded_events, 0);
    bytes[outer.module_handoff_range()].copy_from_slice(old.module_handoff().canonical_bytes());
    let (_, trace) = measure_allocations(|| {
        seal_inert_semantic_compiler_module_handoff_v4(
            outer,
            &mut bytes,
            cap_identity,
            old.module_handoff().identity(),
            |_| Ok::<_, ()>(()),
        )
        .unwrap();
    });
    assert_eq!(trace.sizes, [] as [usize; 0]);
    assert_eq!(trace.unrecorded_events, 0);
    bytes
}

fn assert_native_v4_has_no_payload_copies() {
    use fe2o3_compiler_ffi::{
        InertSemanticCompilerModuleHandoffErrorV4, InertSemanticCompilerModuleHandoffV4,
        MAX_INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_BYTES_V4,
    };
    let mut prior = None;
    for (module, receipt, output, source) in [
        (
            REPRESENTATIVE_MODULE_BYTES,
            REPRESENTATIVE_RECEIPT_BYTES,
            1024 * 1024,
            512 * 1024,
        ),
        (
            MAX_COMPILER_MODULE_BYTES_V1,
            MAX_LINEAGE_RECEIPT_PREIMAGE_BYTES_V3,
            64 * 1024 * 1024,
            4 * 1024 * 1024,
        ),
    ] {
        let bytes = native_v4_wire(module, receipt, output, source);
        let pointer = bytes.as_ptr();
        let (owner, trace) = measure_allocations(|| {
            InertSemanticCompilerModuleHandoffV4::decode_owned(bytes).unwrap()
        });
        assert_eq!(owner.canonical_bytes().as_ptr(), pointer);
        let outer = owner.canonical_bytes();
        for part in [
            owner.capsule().canonical_bytes(),
            owner.capsule().base().canonical_bytes(),
            owner.capsule().carrier_bytes(),
            owner.module_handoff().canonical_bytes(),
            owner.module_handoff().module_bytes(),
            owner.pair_binding_bytes(),
        ] {
            assert_retained_range(outer, part, "native V4 payload");
        }
        for preimage in receipt_preimages(owner.capsule().base()) {
            assert_retained_range(outer, preimage, "native V4 base receipt");
        }
        assert_eq!(trace.unrecorded_events, 0);
        assert!(
            trace
                .sizes
                .iter()
                .all(|size| *size < REPRESENTATIVE_RECEIPT_BYTES),
            "{trace:?}"
        );
        if let Some(prior) = prior {
            assert_eq!(
                trace, prior,
                "native decode topology must not scale with payload length"
            );
        }
        prior = Some(trace);
    }
    let mut bad = native_v4_wire(1024, 512, 128, 64);
    bad[0] ^= 1;
    let (rejected, trace) =
        measure_allocations(|| InertSemanticCompilerModuleHandoffV4::decode_owned(bad));
    assert!(rejected.is_err());
    assert_eq!(trace.sizes, [] as [usize; 0]);
    assert_eq!(trace.unrecorded_events, 0);
    let good = native_v4_wire(1024, 512, 128, 64);
    for (offset, declared, expected) in [
        (
            12,
            MAX_INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_BYTES_V4 + 1,
            InertSemanticCompilerModuleHandoffErrorV3::OuterByteBoundExceeded,
        ),
        (
            24,
            MAX_INERT_PRODUCTION_SEMANTIC_CAPSULE_BYTES_V3 + 1,
            InertSemanticCompilerModuleHandoffErrorV3::CapsuleByteBoundExceeded,
        ),
        (
            32,
            MAX_COMPILER_MODULE_HANDOFF_BYTES_V2 + 1,
            InertSemanticCompilerModuleHandoffErrorV3::ModuleHandoffByteBoundExceeded,
        ),
        (
            24,
            0,
            InertSemanticCompilerModuleHandoffErrorV3::CapsuleByteBoundExceeded,
        ),
        (
            32,
            0,
            InertSemanticCompilerModuleHandoffErrorV3::ModuleHandoffByteBoundExceeded,
        ),
    ] {
        let mut bytes = good.clone();
        bytes[offset..offset + 8].copy_from_slice(&(declared as u64).to_le_bytes());
        let (result, trace) =
            measure_allocations(|| InertSemanticCompilerModuleHandoffV4::decode_owned(bytes));
        assert!(
            matches!(result, Err(InertSemanticCompilerModuleHandoffErrorV4::Framing(error)) if error == expected)
        );
        assert_eq!(trace.sizes, [] as [usize; 0]);
        assert_eq!(trace.unrecorded_events, 0);
    }
}

fn assert_native_carrier_has_no_payload_copies() {
    use fe2o3_compiler_lineage::{
        MAX_NATIVE_REFINED_FORWARDING_CARRIER_STORAGE_V1 as LIMIT,
        MAX_NATIVE_REFINED_FORWARDING_OUTPUT_BYTES_V1 as OUTPUT,
        MAX_NATIVE_REFINED_FORWARDING_SOURCE_BYTES_V1 as SOURCE,
        NativeRefinedForwardingCarrierLayoutV1 as Layout,
        read_native_refined_forwarding_carrier_v1 as read,
        seal_native_refined_forwarding_carrier_v1 as seal,
    };
    for (output, source) in [(19, 7), (OUTPUT, SOURCE)] {
        let layout = Layout::new::<()>(output, source).unwrap();
        let mut bytes = vec![0x37; layout.encoded_len()];
        let (identity, allocations) =
            measure_allocations(|| seal(layout, &mut bytes, LIMIT, |_| Ok::<_, ()>(())).unwrap());
        assert_eq!(allocations.sizes, [] as [usize; 0]);
        assert_eq!(allocations.unrecorded_events, 0);
        let (frame, allocations) =
            measure_allocations(|| read(&bytes, LIMIT, |_| Ok::<_, ()>(())).unwrap());
        assert_eq!(allocations.sizes, [] as [usize; 0]);
        assert_eq!(allocations.unrecorded_events, 0);
        assert_eq!(frame.identity(), identity);
        assert_eq!(
            frame.output().as_ptr(),
            bytes[layout.output_range()].as_ptr()
        );
        assert_eq!(
            frame.source_packet().as_ptr(),
            bytes[layout.source_range()].as_ptr()
        );
    }
}
