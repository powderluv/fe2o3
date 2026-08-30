//! Independent replay of the exact production Kernel IR to LLVM derivation.

use std::error::Error;
use std::fmt;

use dialect_amdgcn::{
    CanonicalProductionKirToLlvmReplayEvidenceV1, ProductionKirToLlvmReplayErrorV1,
    ValidatedProductionKirToLlvmReplayV1,
};
use fe2o3_compiler_lineage::{
    InertAmdgpuLoweringReceiptIdentityV3, InertAmdgpuLoweringReceiptV3,
    InertKernelIrReceiptIdentityV3, InertKernelIrReceiptV3,
};

/// Compiler lineage receipts whose target binding and KIR-to-LLVM lowering were replayed exactly.
///
/// This value remains inert: it authenticates neither compiler origin nor LLVM-to-machine
/// translation and grants no object, publication, load, or launch authority.
#[derive(Debug)]
#[must_use = "dropping validated compiler replay abandons exact KIR-to-LLVM custody"]
pub struct ValidatedCompilerKirToLlvmReplayV1 {
    kernel_ir_receipt_identity: InertKernelIrReceiptIdentityV3,
    amdgpu_lowering_receipt_identity: InertAmdgpuLoweringReceiptIdentityV3,
    replay: ValidatedProductionKirToLlvmReplayV1,
}

impl ValidatedCompilerKirToLlvmReplayV1 {
    pub const fn kernel_ir_receipt_identity(&self) -> InertKernelIrReceiptIdentityV3 {
        self.kernel_ir_receipt_identity
    }

    pub const fn amdgpu_lowering_receipt_identity(&self) -> InertAmdgpuLoweringReceiptIdentityV3 {
        self.amdgpu_lowering_receipt_identity
    }

    pub const fn replay(&self) -> &ValidatedProductionKirToLlvmReplayV1 {
        &self.replay
    }

    pub const fn has_exact_target_binding_replay(&self) -> bool {
        self.replay.has_exact_target_binding_replay()
    }

    pub const fn has_exact_kir_to_llvm_replay(&self) -> bool {
        self.replay.has_exact_kir_to_llvm_replay()
    }

    pub const fn authenticates_compiler_origin(&self) -> bool {
        false
    }

    pub const fn establishes_llvm_to_machine_refinement(&self) -> bool {
        false
    }

    pub const fn grants_runtime_authority(&self) -> bool {
        false
    }
}

/// Independently decodes and replays the KIR-to-LLVM derivation carried by two generic receipts.
pub fn validate_compiler_kir_to_llvm_replay_v1(
    kernel_ir: &InertKernelIrReceiptV3,
    amdgpu_lowering: &InertAmdgpuLoweringReceiptV3,
) -> Result<ValidatedCompilerKirToLlvmReplayV1, CompilerKirToLlvmReplayValidationErrorV1> {
    let evidence =
        CanonicalProductionKirToLlvmReplayEvidenceV1::decode(amdgpu_lowering.canonical_preimage())
            .map_err(CompilerKirToLlvmReplayValidationErrorV1::Replay)?;
    let replay = evidence
        .validate_against_neutral_kernel_ir(kernel_ir.canonical_preimage())
        .map_err(CompilerKirToLlvmReplayValidationErrorV1::Replay)?;
    Ok(ValidatedCompilerKirToLlvmReplayV1 {
        kernel_ir_receipt_identity: kernel_ir.identity(),
        amdgpu_lowering_receipt_identity: amdgpu_lowering.identity(),
        replay,
    })
}

#[derive(Debug)]
pub enum CompilerKirToLlvmReplayValidationErrorV1 {
    Replay(ProductionKirToLlvmReplayErrorV1),
}

impl fmt::Display for CompilerKirToLlvmReplayValidationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Replay(error) => write!(
                formatter,
                "compiler KIR-to-LLVM replay validation failed: {error}"
            ),
        }
    }
}

impl Error for CompilerKirToLlvmReplayValidationErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Replay(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use dialect_amdgcn::{
        bind_production_target_v1, bind_production_upstream_llvm_layout_v1,
        lower_kernel_to_gfx942_xnack_minus_llvm_ir,
    };
    use fe2o3_amd_target::ProductionAmdTargetProfileV1;
    use fe2o3_compiler_lineage::{InertAmdgpuLoweringReceiptV3, InertKernelIrReceiptV3};
    use fe2o3_kernel_ir::{
        BasicBlock, BlockId, Function, Kernel, LaunchDomain, LaunchExtent, Module, Signature,
        Terminator, VerifiedCanonicalKernelIrV8, WorkgroupSize,
    };

    use super::*;

    fn neutral_module(name: &str) -> Module {
        let mut block = BasicBlock::new(BlockId(0));
        block.terminator = Some(Terminator::Return { values: vec![] });
        let function = Function::kernel_entry(
            format!("{name}_entry"),
            Signature::new(vec![], vec![]),
            vec![],
            vec![block],
        );
        let mut kernel = Kernel::new(
            format!("{name}_kernel"),
            format!("{name}_entry"),
            LaunchDomain::D1 {
                x: LaunchExtent::Static(1),
            },
        );
        kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
        let mut module = Module::new(name);
        module.functions.push(function);
        module.kernels.push(kernel);
        module
    }

    fn receipts(name: &str) -> (InertKernelIrReceiptV3, InertAmdgpuLoweringReceiptV3) {
        let neutral_module = neutral_module(name);
        let neutral = VerifiedCanonicalKernelIrV8::from_module(neutral_module.clone()).unwrap();
        let neutral_bytes = neutral.canonical_bytes();
        let target =
            bind_production_target_v1(&neutral_module, ProductionAmdTargetProfileV1::Gfx942)
                .unwrap();
        let dialect =
            lower_kernel_to_gfx942_xnack_minus_llvm_ir(target.module(), target.kernel_id())
                .unwrap();
        let llvm = bind_production_upstream_llvm_layout_v1(&dialect).unwrap();
        let evidence = CanonicalProductionKirToLlvmReplayEvidenceV1::from_live_inputs(
            neutral_bytes,
            target.module(),
            ProductionAmdTargetProfileV1::Gfx942,
            &llvm,
        )
        .unwrap();
        (
            InertKernelIrReceiptV3::from_canonical_preimage(neutral_bytes).unwrap(),
            InertAmdgpuLoweringReceiptV3::from_canonical_preimage(evidence.canonical_bytes())
                .unwrap(),
        )
    }

    #[test]
    fn generic_receipts_independently_replay_exact_target_kir_and_llvm() {
        let (kernel_ir, amdgpu) = receipts("independent");
        let validated = validate_compiler_kir_to_llvm_replay_v1(&kernel_ir, &amdgpu).unwrap();
        assert_eq!(validated.kernel_ir_receipt_identity(), kernel_ir.identity());
        assert_eq!(
            validated.amdgpu_lowering_receipt_identity(),
            amdgpu.identity()
        );
        assert!(validated.has_exact_target_binding_replay());
        assert!(validated.has_exact_kir_to_llvm_replay());
        assert!(!validated.authenticates_compiler_origin());
        assert!(!validated.establishes_llvm_to_machine_refinement());
        assert!(!validated.grants_runtime_authority());
    }

    #[test]
    fn valid_receipts_from_different_compilations_cannot_be_spliced() {
        let (kernel_ir_a, _) = receipts("first");
        let (_, amdgpu_b) = receipts("second");
        assert!(matches!(
            validate_compiler_kir_to_llvm_replay_v1(&kernel_ir_a, &amdgpu_b),
            Err(CompilerKirToLlvmReplayValidationErrorV1::Replay(
                ProductionKirToLlvmReplayErrorV1::IdentityMismatch {
                    field: "neutral Kernel IR"
                }
            ))
        ));
    }

    #[test]
    fn legacy_association_payload_is_not_misreported_as_replay_evidence() {
        let (kernel_ir, _) = receipts("legacy");
        let legacy = InertAmdgpuLoweringReceiptV3::from_canonical_preimage(
            b"association-only legacy transcript",
        )
        .unwrap();
        assert!(matches!(
            validate_compiler_kir_to_llvm_replay_v1(&kernel_ir, &legacy),
            Err(CompilerKirToLlvmReplayValidationErrorV1::Replay(
                ProductionKirToLlvmReplayErrorV1::Truncated
                    | ProductionKirToLlvmReplayErrorV1::InvalidHeader
            ))
        ));
    }
}
