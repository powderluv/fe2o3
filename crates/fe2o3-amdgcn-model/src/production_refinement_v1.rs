use std::error::Error;
use std::fmt;

use fe2o3_amd_target::{PRODUCTION_AMDHSA_LLVM_DATA_LAYOUT_V1, ProductionAmdTargetProfileV1};
use fe2o3_kernel_ir::{
    FunctionId, KernelId, Module, TargetCapability, VerificationErrors, WaveWidth,
    gfx942_xnack_minus_target_capability, gfx950_xnack_minus_target_capability, verify_module,
};

use crate::{AMDGPU_TRIPLE, GFX942_XNACK_MINUS_DATA_LAYOUT};

/// The deterministic target-bound Kernel IR produced from one neutral module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionTargetBoundKernelIrV1 {
    profile: ProductionAmdTargetProfileV1,
    module: Module,
    kernel_id: KernelId,
}

impl ProductionTargetBoundKernelIrV1 {
    pub const fn profile(&self) -> ProductionAmdTargetProfileV1 {
        self.profile
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    pub fn into_parts(self) -> (Module, KernelId) {
        (self.module, self.kernel_id)
    }
}

/// Closed failures for the production neutral-KIR target-binding transform.
#[derive(Debug)]
pub enum ProductionTargetBindingErrorV1 {
    KernelClosure { observed: usize },
    MissingEntry { entry: FunctionId },
    InvalidTargetBoundModule(VerificationErrors),
}

impl fmt::Display for ProductionTargetBindingErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelClosure { observed } => write!(
                formatter,
                "production target binding requires exactly one kernel, observed {observed}"
            ),
            Self::MissingEntry { entry } => write!(
                formatter,
                "production target binding cannot find kernel entry {entry}"
            ),
            Self::InvalidTargetBoundModule(error) => {
                write!(
                    formatter,
                    "production target-bound Kernel IR is invalid: {error}"
                )
            }
        }
    }
}

impl Error for ProductionTargetBindingErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidTargetBoundModule(error) => Some(error),
            Self::KernelClosure { .. } | Self::MissingEntry { .. } => None,
        }
    }
}

/// Applies the sole production target transform to a target-neutral Kernel IR module.
///
/// The transform adds only the exact processor and Wave64 requirements to the
/// module, its sole kernel, and that kernel's entry function. It then verifies
/// the complete result before returning target-bound custody.
pub fn bind_production_target_v1(
    neutral_module: &Module,
    profile: ProductionAmdTargetProfileV1,
) -> Result<ProductionTargetBoundKernelIrV1, ProductionTargetBindingErrorV1> {
    let mut module = neutral_module.clone();
    let target = match profile {
        ProductionAmdTargetProfileV1::Gfx942 => gfx942_xnack_minus_target_capability(),
        ProductionAmdTargetProfileV1::Gfx950 => gfx950_xnack_minus_target_capability(),
    };
    let wave = TargetCapability::WaveWidth(WaveWidth::Wave64);

    module.required_capabilities.insert(target.clone());
    module.required_capabilities.insert(wave.clone());

    let observed = module.kernels.len();
    let [kernel] = module.kernels.as_mut_slice() else {
        return Err(ProductionTargetBindingErrorV1::KernelClosure { observed });
    };
    kernel.required_capabilities.insert(target.clone());
    kernel.required_capabilities.insert(wave.clone());
    let kernel_id = kernel.id.clone();
    let entry_id = kernel.entry.clone();

    let entry = module
        .functions
        .iter_mut()
        .find(|function| function.id == entry_id)
        .ok_or_else(|| ProductionTargetBindingErrorV1::MissingEntry {
            entry: entry_id.clone(),
        })?;
    entry.required_capabilities.insert(target);
    entry.required_capabilities.insert(wave);

    verify_module(&module).map_err(ProductionTargetBindingErrorV1::InvalidTargetBoundModule)?;
    Ok(ProductionTargetBoundKernelIrV1 {
        profile,
        module,
        kernel_id,
    })
}

/// Closed failures for exact production LLVM target-header binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionLlvmLayoutBindingErrorV1 {
    NonCanonicalTargetHeader,
}

impl fmt::Display for ProductionLlvmLayoutBindingErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCanonicalTargetHeader => formatter
                .write_str("verified AMDGPU lowering did not retain one canonical target header"),
        }
    }
}

impl Error for ProductionLlvmLayoutBindingErrorV1 {}

/// Rebinds deterministic dialect LLVM to the layout measured from upstream LLVM.
///
/// The input must contain exactly one canonical AMDGPU target header. The
/// returned text is suitable for exact replay by an independent verifier; it
/// does not grant object-generation, linking, publication, or execution authority.
pub fn bind_production_upstream_llvm_layout_v1(
    dialect_llvm_ir: &str,
) -> Result<String, ProductionLlvmLayoutBindingErrorV1> {
    let triple_header = format!("target triple = \"{AMDGPU_TRIPLE}\"\n");
    let dialect_layout = format!("target datalayout = \"{GFX942_XNACK_MINUS_DATA_LAYOUT}\"\n");
    let expected_prefix = format!("{triple_header}{dialect_layout}\n");
    if !dialect_llvm_ir.starts_with(&expected_prefix)
        || dialect_llvm_ir.matches("target triple =").count() != 1
        || dialect_llvm_ir.matches("target datalayout =").count() != 1
    {
        return Err(ProductionLlvmLayoutBindingErrorV1::NonCanonicalTargetHeader);
    }

    let mut bound = String::with_capacity(
        dialect_llvm_ir.len()
            + PRODUCTION_AMDHSA_LLVM_DATA_LAYOUT_V1
                .len()
                .saturating_sub(GFX942_XNACK_MINUS_DATA_LAYOUT.len()),
    );
    bound.push_str(&triple_header);
    bound.push_str("target datalayout = \"");
    bound.push_str(PRODUCTION_AMDHSA_LLVM_DATA_LAYOUT_V1);
    bound.push_str("\"\n\n");
    bound.push_str(&dialect_llvm_ir[expected_prefix.len()..]);
    Ok(bound)
}

#[cfg(test)]
mod tests {
    use fe2o3_kernel_ir::{
        BasicBlock, BlockId, Function, LaunchDomain, LaunchExtent, Module, Signature, Terminator,
        WorkgroupSize, gfx942_xnack_minus_target_capability,
    };

    use super::*;

    fn neutral_module() -> Module {
        let mut block = BasicBlock::new(BlockId(0));
        block.terminator = Some(Terminator::Return { values: vec![] });
        let function =
            Function::kernel_entry("entry", Signature::new(vec![], vec![]), vec![], vec![block]);
        let mut kernel = fe2o3_kernel_ir::Kernel::new(
            "kernel",
            "entry",
            LaunchDomain::D1 {
                x: LaunchExtent::Static(1),
            },
        );
        kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
        let mut module = Module::new("production-refinement-test");
        module.functions.push(function);
        module.kernels.push(kernel);
        module
    }

    #[test]
    fn target_binding_is_exact_and_does_not_mutate_neutral_input() {
        let neutral = neutral_module();
        let bound = bind_production_target_v1(&neutral, ProductionAmdTargetProfileV1::Gfx942)
            .expect("target binding succeeds");
        let target = gfx942_xnack_minus_target_capability();
        let wave = TargetCapability::WaveWidth(WaveWidth::Wave64);

        assert!(neutral.required_capabilities.is_empty());
        assert_eq!(bound.profile(), ProductionAmdTargetProfileV1::Gfx942);
        assert_eq!(bound.kernel_id(), &KernelId::new("kernel"));
        assert!(bound.module().required_capabilities.contains(&target));
        assert!(bound.module().required_capabilities.contains(&wave));
        assert!(
            bound.module().kernels[0]
                .required_capabilities
                .contains(&target)
        );
        assert!(
            bound.module().kernels[0]
                .required_capabilities
                .contains(&wave)
        );
        assert!(
            bound.module().functions[0]
                .required_capabilities
                .contains(&target)
        );
        assert!(
            bound.module().functions[0]
                .required_capabilities
                .contains(&wave)
        );
    }

    #[test]
    fn target_binding_rejects_non_singleton_kernel_closure() {
        let mut neutral = neutral_module();
        neutral.kernels.clear();
        assert!(matches!(
            bind_production_target_v1(&neutral, ProductionAmdTargetProfileV1::Gfx942),
            Err(ProductionTargetBindingErrorV1::KernelClosure { observed: 0 })
        ));

        let mut neutral = neutral_module();
        neutral.kernels.push(neutral.kernels[0].clone());
        assert!(matches!(
            bind_production_target_v1(&neutral, ProductionAmdTargetProfileV1::Gfx942),
            Err(ProductionTargetBindingErrorV1::KernelClosure { observed: 2 })
        ));
    }

    #[test]
    fn llvm_layout_binding_requires_the_exact_unique_header() {
        let dialect = format!(
            "target triple = \"{AMDGPU_TRIPLE}\"\ntarget datalayout = \"{GFX942_XNACK_MINUS_DATA_LAYOUT}\"\n\ndefine void @kernel() {{\n  ret void\n}}\n"
        );
        let expected = format!(
            "target triple = \"{AMDGPU_TRIPLE}\"\ntarget datalayout = \"{PRODUCTION_AMDHSA_LLVM_DATA_LAYOUT_V1}\"\n\ndefine void @kernel() {{\n  ret void\n}}\n"
        );
        assert_eq!(
            bind_production_upstream_llvm_layout_v1(&dialect).unwrap(),
            expected
        );

        for hostile in [
            dialect.replacen("target triple", "source triple", 1),
            dialect.replacen("target datalayout", "source datalayout", 1),
            format!("{dialect}target triple = \"{AMDGPU_TRIPLE}\"\n"),
            format!("{dialect}target datalayout = \"{GFX942_XNACK_MINUS_DATA_LAYOUT}\"\n"),
        ] {
            assert_eq!(
                bind_production_upstream_llvm_layout_v1(&hostile),
                Err(ProductionLlvmLayoutBindingErrorV1::NonCanonicalTargetHeader)
            );
        }
    }
}
