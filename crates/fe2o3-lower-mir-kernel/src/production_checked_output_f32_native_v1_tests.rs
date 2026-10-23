use super::*;
use std::{fmt::Write, ptr};

use dialect_amdgcn::{
    NativeV12TextDescriptorReplayErrorV1 as NativeError, bind_production_llvm22_worker_layout_v1,
    check_native_v12_text_descriptor_relation_v1 as check,
    lower_canonical_v12_compiler_module_to_gfx942_xnack_minus_llvm_ir_with_semantic_anchors_v1 as lower_942,
    lower_canonical_v12_compiler_module_to_gfx950_xnack_minus_llvm_ir_with_semantic_anchors_v1 as lower_950,
};
use fe2o3_kernel_descriptor::{
    BlockSizeV1, BuildEvidenceV1, CanonicalCodeObjectDigest, CodeObjectVersion, CompilerIdentityV1,
    DeviceDescriptorTableV1 as Table, DeviceLayoutDescriptorV1, DeviceLayoutRecordV1,
    DeviceTargetV1, DimensionsV1, EvidenceDigest, EvidenceIdentity, KernelAbiLayoutV1,
    KernelDescriptorV1, KernelId as DescriptorId, LaunchConstraintsV1, LogicalArgumentV1,
    ProducerIdentityV1, ScalarTypeV1, SourceTypeDescriptorV1, SourceTypeRecordV1, Text, ValidName,
    encode_device_descriptor_table_v1,
};
use fe2o3_kernel_ir::InertCanonicalKernelIrContractCatalogV1 as Catalog;

// Independent inert descriptor inputs. They model this fixture's three actual
// physical arguments, but do not claim source-derived descriptor authority.
pub(super) fn table(profile: Profile) -> Table {
    let pointer = SourceTypeRecordV1::new(SourceTypeDescriptorV1::global_mut_pointer(
        ScalarTypeV1::F32,
    ));
    let scalar = SourceTypeRecordV1::new(SourceTypeDescriptorV1::scalar(ScalarTypeV1::F32));
    let pointer_layout = DeviceLayoutRecordV1::new(DeviceLayoutDescriptorV1::global_mut_pointer(
        ScalarTypeV1::F32,
    ));
    let scalar_layout =
        DeviceLayoutRecordV1::new(DeviceLayoutDescriptorV1::scalar(ScalarTypeV1::F32));
    let arguments = vec![
        LogicalArgumentV1::global_mut_pointer(
            0,
            ValidName::new("output").unwrap(),
            &pointer,
            &pointer_layout,
            0,
        )
        .unwrap(),
        LogicalArgumentV1::scalar(
            1,
            ValidName::new("left").unwrap(),
            &scalar,
            &scalar_layout,
            8,
        )
        .unwrap(),
        LogicalArgumentV1::scalar(
            2,
            ValidName::new("right").unwrap(),
            &scalar,
            &scalar_layout,
            12,
        )
        .unwrap(),
    ];
    let evidence = BuildEvidenceV1::new(
        EvidenceIdentity::from_opaque_bytes([1; 32]),
        EvidenceDigest::from_sha256_bytes([2; 32]),
    );
    let kernel = KernelDescriptorV1::new(
        DescriptorId::from_bytes([1; 32]),
        ValidName::new("logical_0").unwrap(),
        ValidName::new(FP_NAME).unwrap(),
        ValidName::new(format!("{FP_NAME}.kd")).unwrap(),
        evidence,
        evidence,
        vec![],
        KernelAbiLayoutV1::new(16, 16, 8).unwrap(),
        LaunchConstraintsV1::new(
            1,
            BlockSizeV1::Exact(DimensionsV1::new(1, 1, 1).unwrap()),
            DimensionsV1::new(1, 1, 1).unwrap(),
            64,
            0,
            0,
        )
        .unwrap(),
        arguments,
    )
    .unwrap();
    Table::new(
        CanonicalCodeObjectDigest::from_bytes([0; 32]),
        CodeObjectVersion::V6,
        CompilerIdentityV1::new(
            Text::new("inert-component").unwrap(),
            Text::new("1").unwrap(),
            [0; 20],
        ),
        ProducerIdentityV1::new(
            Text::new("inert-component").unwrap(),
            Text::new("1").unwrap(),
        ),
        DeviceTargetV1::parse(profile.device_target()).unwrap(),
        vec![pointer, scalar],
        vec![pointer_layout, scalar_layout],
        vec![kernel],
    )
    .unwrap()
}

pub(super) fn final_text(prefix: &str, descriptors: &Table) -> String {
    let bytes = encode_device_descriptor_table_v1(descriptors).unwrap();
    let mut text = prefix.to_owned();
    text.push_str(
        "\nmodule asm \".section .fe2o3.kd.v1,\\22\\22,@progbits\"\nmodule asm \".balign 8\"\n",
    );
    for chunk in bytes.chunks(16) {
        text.push_str("module asm \".byte ");
        for (ordinal, byte) in chunk.iter().enumerate() {
            if ordinal != 0 {
                text.push_str(", ");
            }
            write!(text, "0x{byte:02x}").unwrap();
        }
        text.push_str("\"\n");
    }
    text
}

pub(super) fn check_actual(
    owner: &crate::ProductionCheckedOutputOwnerPolicy4V1,
    profile: Profile,
    recipe: FpRecipe,
    budget: &mut AssertOriginBudgetV1<'_>,
) {
    let entry = budget.storage();
    // Test descriptor/text construction uses a conservative bounded fixture
    // envelope, separately from actual catalog and replay-header receipts.
    const FIXTURE: usize = 1 << 20;
    budget.reserve_storage(FIXTURE).unwrap();
    let sha = *owner
        .source_semantic_kir()
        .semantic()
        .semantic()
        .semantic_sha256()
        .as_bytes();
    let (catalog, receipt) = Catalog::from_rows_with_budget(sha, &[], &[], budget).unwrap();
    budget.reserve_storage(receipt.retained_storage()).unwrap();
    let output = owner.output();
    let native = match profile {
        Profile::Gfx942 => lower_942(output),
        Profile::Gfx950 => lower_950(output),
    }
    .unwrap();
    let prefix = bind_production_llvm22_worker_layout_v1(&native).unwrap();
    let opcode = match recipe {
        FpRecipe::Negate => "fneg",
        FpRecipe::Divide => "fdiv",
        FpRecipe::Remainder => unreachable!(),
    };
    assert!(prefix.contains(&format!(" = {opcode} float ")), "{prefix}");
    for attribute in [
        "\"denormal-fp-math-f32\"=\"ieee,ieee\"",
        "\"unsafe-fp-math\"=\"false\"",
        "\"no-infs-fp-math\"=\"false\"",
        "\"no-nans-fp-math\"=\"false\"",
        "\"no-signed-zeros-fp-math\"=\"false\"",
        "\"fp-contract\"=\"off\"",
    ] {
        assert!(
            prefix.contains(attribute),
            "missing strict attribute: {attribute}"
        );
    }
    for flag in [
        " fast ",
        " reassoc ",
        " arcp ",
        " afn ",
        " nnan ",
        " ninf ",
        " nsz ",
        " contract ",
    ] {
        assert!(!prefix.contains(flag), "unexpected fast-math flag: {flag}");
    }
    let descriptors = table(profile);
    let text = final_text(&prefix, &descriptors);
    assert!(native.len() + prefix.len() + text.len() * 3 < FIXTURE);
    let floor = budget.storage();
    let retained = {
        let relation = check(
            output,
            &catalog,
            output.canonical().canonical_bytes(),
            profile,
            &descriptors,
            &text,
            budget,
        )
        .unwrap();
        assert_eq!(budget.storage(), floor);
        let retained = relation.storage().retained_storage();
        budget.reserve_storage(retained).unwrap();
        assert!(ptr::eq(relation.output(), output));
        assert!(ptr::eq(relation.catalog(), &catalog));
        assert!(ptr::eq(relation.descriptors(), &descriptors));
        assert!(ptr::eq(relation.final_llvm(), text.as_str()));
        assert_eq!(relation.pre_descriptor_llvm(), prefix);
        assert_eq!(relation.profile(), profile);
        assert!(!relation.grants_authority());
        retained
    };
    budget.release_storage(retained).unwrap();
    let bytes = output.canonical().canonical_bytes();
    assert!(matches!(
        check(
            output,
            &catalog,
            &bytes[..bytes.len() - 1],
            profile,
            &descriptors,
            &text,
            budget
        ),
        Err(NativeError::OutputBytes)
    ));
    let foreign_profile = match profile {
        Profile::Gfx942 => Profile::Gfx950,
        Profile::Gfx950 => Profile::Gfx942,
    };
    assert!(
        check(
            output,
            &catalog,
            bytes,
            foreign_profile,
            &descriptors,
            &text,
            budget
        )
        .is_err()
    );
    let fast = text.replacen(
        &format!(" = {opcode} float "),
        &format!(" = {opcode} fast float "),
        1,
    );
    assert_ne!(fast, text);
    assert!(
        check(
            output,
            &catalog,
            bytes,
            profile,
            &descriptors,
            &fast,
            budget
        )
        .is_err()
    );
    let altered = text.replacen(&format!(" = {opcode} float "), " = fadd float ", 1);
    assert_ne!(altered, text);
    assert!(
        check(
            output,
            &catalog,
            bytes,
            profile,
            &descriptors,
            &altered,
            budget
        )
        .is_err()
    );
    assert_eq!(budget.storage(), floor);
    drop((native, prefix, descriptors, text, fast, altered));
    drop(catalog);
    budget.release_storage(receipt.retained_storage()).unwrap();
    budget.release_storage(FIXTURE).unwrap();
    assert_eq!(budget.storage(), entry);
}

// Source and Policy4 admission still require the exact original payload.
// Native emission uses an equal-width bitcast, not NaN-class equivalence.
pub(super) fn check_actual_literal_nan(
    owner: &crate::ProductionCheckedOutputOwnerPolicy4V1,
    profile: Profile,
) {
    let output = owner.output();
    let mut literal_count = 0;
    for operation in output
        .module()
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref())
        .flat_map(|body| &body.blocks)
        .flat_map(|block| &block.operations)
    {
        if let OperationKind::Constant(fe2o3_kernel_ir::Constant::F32Bits(bits)) = &operation.kind
            && f32::from_bits(*bits).is_nan()
        {
            assert_eq!(*bits, 0x7fc0_0042);
            literal_count += 1;
        }
    }
    assert_eq!(
        literal_count, 1,
        "actual O must retain the source NaN literal"
    );
    let llvm = match profile {
        Profile::Gfx942 => lower_942(output),
        Profile::Gfx950 => lower_950(output),
    }
    .expect("the admitted literal NaN has an exact LLVM bitcast encoding");
    let llvm = bind_production_llvm22_worker_layout_v1(&llvm).unwrap();
    assert!(llvm.contains("bitcast (i32 2143289410 to float)"));
}
