use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;

use fe2o3_kernel_descriptor::Gfx942LaunchBoundsV1;
use fe2o3_kernel_ir::{
    AMDGPU_DIAGNOSTICS_CAPABILITY_NAME, AMDGPU_DIAGNOSTICS_CAPABILITY_NAMESPACE,
    AMDGPU_EXACT_TARGET_CAPABILITY_NAMESPACE, AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAME,
    AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAMESPACE, AMDGPU_GFX942_INLINE_ASSEMBLY_CAPABILITY_NAME,
    AMDGPU_GFX942_INLINE_ASSEMBLY_CAPABILITY_NAMESPACE,
    AMDGPU_GFX942_XNACK_MINUS_TARGET_CAPABILITY_NAME,
    AMDGPU_GFX950_XNACK_MINUS_TARGET_CAPABILITY_NAME, AddressSpace as KernelAddressSpace,
    AmdGpuDiagnosticOperation, AssemblyConstraint, AssemblyOperandKind, AssemblyOption, Atomic,
    AtomicKind, Axis, BF16_F32_M16N16K16_CAPABILITY, BasicBlock, BinaryOp, BlockId, CastKind,
    CheckedBinaryOperator, ComparePredicate, Constant,
    DiagnosticCode as VerificationDiagnosticCode, F32MathFunction, F32MathImplementation,
    FloatConversionKind, FloatOperation, Function, FunctionId, FunctionRole,
    Gfx950LdsTransposeFormatV1, Gfx950LdsTransposeOperationKindV1, Gfx950LdsTransposeOperationV1,
    IndexKind, IndexedControlFlow, InlineAssembly, InlineAssemblyTarget, IntrinsicKind, Kernel,
    KernelId, LDS_TILE_16X16_XOR4_CAPABILITY, LaunchDomain, LaunchExtent,
    MATRIX_CAPABILITY_NAMESPACE, MATRIX_PROJECTED_KERNARG_POLICY_NAMESPACE_V1,
    MATRIX_SOURCE_ABI_OBSERVATION_NAMESPACE_V2, MatrixElement, MatrixFrontendBindingV2,
    MatrixMultiplyProfile, MatrixOperation, MatrixOperationKind, MatrixProjectedKernargPolicyV1,
    MemoryElementType, MemoryIntrinsicOperation, MemoryOrdering, Module, ModuleId,
    NarrowFloatFormat, Operation, OperationKind, PointerDistanceContract, PointerDistanceKind,
    PointerDistanceUnit, SCALED_FP4_E2M1_F32_M16N16K128_CAPABILITY,
    SCALED_FP4_E2M1_FP8_E4M3_F32_M16N16K128_CAPABILITY, SCALED_FP8_E4M3_F32_M16N16K128_CAPABILITY,
    ScalarType, Signature, SynchronizationScope, TargetCapability, TensorInstructionProfileV1,
    Terminator, Type, UnaryOp, ValueId, VerificationErrors, WaveF32ReductionKindV1, WaveOperation,
    WaveOperationKind, WaveWidth, WidenedFloatBinaryOp, WorkgroupMemoryExtent, WorkgroupSize,
    analyze_control_flow, verify_module,
};

use crate::{AMDGPU_TRIPLE, AmdgcnIntrinsic, Dim};

const MAX_G1_FLAT_WORKGROUP_SIZE: u32 = 1024;
const MAX_COMPILER_MODULE_GRAPH_FUNCTIONS: usize = 1_024;
const MAX_COMPILER_MODULE_GRAPH_KERNELS: usize = 256;
const MAX_COMPILER_MODULE_CALL_EDGES: usize = 131_072;
/// Maximum textual LLVM bytes returned by compiler-module construction.
pub const MAX_COMPILER_MODULE_TEXT_BYTES: usize = 16 * 1024 * 1024;

/// Exact data layout measured from the pinned upstream LLVM target machine.
pub const GFX942_XNACK_MINUS_DATA_LAYOUT: &str =
    fe2o3_amd_target::PRODUCTION_AMDHSA_LLVM_DATA_LAYOUT_V1;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoweringTarget {
    Baseline,
    Gfx942StrictFloatV1,
    Gfx942XnackMinusV1,
    Gfx950XnackMinusV1,
}

impl LoweringTarget {
    const fn requires_physical_workgroup_barrier(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    const fn supports_narrow_float(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    const fn supports_gfx942_inline_assembly(self) -> bool {
        !matches!(self, Self::Baseline | Self::Gfx950XnackMinusV1)
    }

    const fn supports_bf16_matrix_mfma(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    const fn supports_gfx942_matrix_lds(self) -> bool {
        !matches!(self, Self::Baseline | Self::Gfx950XnackMinusV1)
    }

    const fn supports_gfx950_scaled_matrix(self) -> bool {
        matches!(self, Self::Gfx950XnackMinusV1)
    }

    const fn supports_gfx950_collectives_and_lds_transpose(self) -> bool {
        matches!(self, Self::Gfx950XnackMinusV1)
    }

    const fn supports_gfx942_diagnostics(self) -> bool {
        !matches!(self, Self::Baseline | Self::Gfx950XnackMinusV1)
    }

    const fn supports_amdgpu_diagnostics(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    const fn supports_gfx942_xnack_minus_binding(self) -> bool {
        matches!(self, Self::Gfx942XnackMinusV1)
    }

    const fn exact_target_binding(self) -> Option<&'static str> {
        match self {
            Self::Gfx942XnackMinusV1 => Some(AMDGPU_GFX942_XNACK_MINUS_TARGET_CAPABILITY_NAME),
            Self::Gfx950XnackMinusV1 => Some(AMDGPU_GFX950_XNACK_MINUS_TARGET_CAPABILITY_NAME),
            _ => None,
        }
    }

    const fn llvm_function_attributes(self) -> &'static str {
        match self {
            Self::Baseline => "",
            Self::Gfx942StrictFloatV1 => {
                " \"target-cpu\"=\"gfx942\" \"denormal-fp-math-f32\"=\"ieee,ieee\" \"unsafe-fp-math\"=\"false\" \"no-infs-fp-math\"=\"false\" \"no-nans-fp-math\"=\"false\" \"no-signed-zeros-fp-math\"=\"false\" \"approx-func-fp-math\"=\"false\""
            }
            Self::Gfx942XnackMinusV1 => {
                " \"target-cpu\"=\"gfx942\" \"denormal-fp-math-f32\"=\"ieee,ieee\" \"unsafe-fp-math\"=\"false\" \"no-infs-fp-math\"=\"false\" \"no-nans-fp-math\"=\"false\" \"no-signed-zeros-fp-math\"=\"false\" \"approx-func-fp-math\"=\"false\" \"fp-contract\"=\"off\""
            }
            Self::Gfx950XnackMinusV1 => {
                " \"target-cpu\"=\"gfx950\" \"denormal-fp-math-f32\"=\"ieee,ieee\" \"unsafe-fp-math\"=\"false\" \"no-infs-fp-math\"=\"false\" \"no-nans-fp-math\"=\"false\" \"no-signed-zeros-fp-math\"=\"false\" \"approx-func-fp-math\"=\"false\" \"fp-contract\"=\"off\""
            }
        }
    }

    const fn data_layout(self) -> Option<&'static str> {
        match self {
            Self::Gfx942XnackMinusV1 | Self::Gfx950XnackMinusV1 => {
                Some(GFX942_XNACK_MINUS_DATA_LAYOUT)
            }
            Self::Baseline | Self::Gfx942StrictFloatV1 => None,
        }
    }

    const fn wave_target_feature(self, width: WaveWidth) -> &'static str {
        match (self, width) {
            (Self::Gfx942XnackMinusV1 | Self::Gfx950XnackMinusV1, WaveWidth::Wave32) => {
                " \"target-features\"=\"+wavefrontsize32,-wavefrontsize64,-xnack\""
            }
            (Self::Gfx942XnackMinusV1 | Self::Gfx950XnackMinusV1, WaveWidth::Wave64) => {
                " \"target-features\"=\"-wavefrontsize32,+wavefrontsize64,-xnack\""
            }
            (_, WaveWidth::Wave32) => " \"target-features\"=\"+wavefrontsize32,-wavefrontsize64\"",
            (_, WaveWidth::Wave64) => " \"target-features\"=\"-wavefrontsize32,+wavefrontsize64\"",
        }
    }
}

#[derive(Clone, Copy)]
struct Gfx942AssemblyInstruction {
    mnemonic: &'static str,
    constraint: AssemblyConstraint,
    input_count: usize,
}

fn gfx942_assembly_instruction(mnemonic: &str) -> Option<Gfx942AssemblyInstruction> {
    let (mnemonic, constraint, input_count) = match mnemonic {
        "v_mov_b32" => ("v_mov_b32", AssemblyConstraint::Vgpr32, 1),
        "s_mov_b32" => ("s_mov_b32", AssemblyConstraint::Sgpr32, 1),
        "v_add_u32" => ("v_add_u32", AssemblyConstraint::Vgpr32, 2),
        "v_sub_u32" => ("v_sub_u32", AssemblyConstraint::Vgpr32, 2),
        "v_and_b32" => ("v_and_b32", AssemblyConstraint::Vgpr32, 2),
        "v_or_b32" => ("v_or_b32", AssemblyConstraint::Vgpr32, 2),
        "v_xor_b32" => ("v_xor_b32", AssemblyConstraint::Vgpr32, 2),
        _ => return None,
    };
    Some(Gfx942AssemblyInstruction {
        mnemonic,
        constraint,
        input_count,
    })
}

/// Stable rejection categories for the first target-neutral AMDGPU lowering slice.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LoweringDiagnosticCode {
    InputVerification(VerificationDiagnosticCode),
    MissingKernel,
    AmbiguousKernel,
    ConflictingSymbol,
    ResourceLimit,
    IncompatibleWaveCallGraph,
    MissingWaveWidth,
    UnsafeSymbolName,
    UnsupportedLaunchDomain,
    MissingWorkgroupSize,
    UnsupportedWorkgroupSize,
    InvalidLaunchPolicy,
    UnsupportedCapability,
    KernelEntryDeclaration,
    UnsupportedResults,
    UnsupportedParameter,
    UnsupportedType,
    UnsupportedAddressSpace,
    UnsupportedBlockArguments,
    UnsupportedOperation,
    UnsupportedAtomic,
    UnsupportedBarrier,
    UnprovenBarrierConvergence,
    UnsupportedWorkgroupMemory,
    UnsupportedWaveOperation,
    UnsupportedFloatOperation,
    UnsupportedDiagnosticOperation,
    UnsupportedInlineAssembly,
    UnsupportedMatrixOperation,
    UnsupportedAssemblyInstruction,
    AssemblyOperandMismatch,
    AssemblyEffectMismatch,
    UnsupportedCast,
    UnsupportedConstant,
    UnsupportedTerminator,
    IrreducibleControlFlow,
}

/// A deterministic source location in the kernel IR.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LoweringLocation {
    pub module: ModuleId,
    pub kernel: Option<KernelId>,
    pub function: Option<FunctionId>,
    pub block: Option<BlockId>,
    pub operation: Option<usize>,
}

impl LoweringLocation {
    fn module(module: &Module) -> Self {
        Self {
            module: module.id.clone(),
            kernel: None,
            function: None,
            block: None,
            operation: None,
        }
    }

    fn kernel(module: &Module, kernel: &Kernel) -> Self {
        Self {
            kernel: Some(kernel.id.clone()),
            ..Self::module(module)
        }
    }

    fn function(module: &Module, kernel: &Kernel, function: &Function) -> Self {
        Self {
            function: Some(function.id.clone()),
            ..Self::kernel(module, kernel)
        }
    }

    fn device_function(module: &Module, function: &Function) -> Self {
        Self {
            function: Some(function.id.clone()),
            ..Self::module(module)
        }
    }

    fn block(module: &Module, kernel: &Kernel, function: &Function, block: BlockId) -> Self {
        Self {
            block: Some(block),
            ..Self::function(module, kernel, function)
        }
    }

    fn operation(
        module: &Module,
        kernel: &Kernel,
        function: &Function,
        block: BlockId,
        operation: usize,
    ) -> Self {
        Self {
            operation: Some(operation),
            ..Self::block(module, kernel, function, block)
        }
    }

    fn device_block(module: &Module, function: &Function, block: BlockId) -> Self {
        Self {
            block: Some(block),
            ..Self::device_function(module, function)
        }
    }

    fn device_operation(
        module: &Module,
        function: &Function,
        block: BlockId,
        operation: usize,
    ) -> Self {
        Self {
            operation: Some(operation),
            ..Self::device_block(module, function, block)
        }
    }
}

impl fmt::Display for LoweringLocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "module {}", self.module)?;
        if let Some(kernel) = &self.kernel {
            write!(formatter, ", kernel {kernel}")?;
        }
        if let Some(function) = &self.function {
            write!(formatter, ", function {function}")?;
        }
        if let Some(block) = self.block {
            write!(formatter, ", {block}")?;
        }
        if let Some(operation) = self.operation {
            write!(formatter, ", op {operation}")?;
        }
        Ok(())
    }
}

/// One stable, located lowering diagnostic.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LoweringDiagnostic {
    pub location: LoweringLocation,
    pub code: LoweringDiagnosticCode,
    pub message: String,
}

impl fmt::Display for LoweringDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: {:?}: {}",
            self.location, self.code, self.message
        )
    }
}

/// A deterministic set of errors produced before textual emission succeeds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoweringErrors {
    diagnostics: Vec<LoweringDiagnostic>,
}

impl LoweringErrors {
    pub fn diagnostics(&self) -> &[LoweringDiagnostic] {
        &self.diagnostics
    }

    pub fn into_diagnostics(self) -> Vec<LoweringDiagnostic> {
        self.diagnostics
    }

    pub fn contains(&self, code: LoweringDiagnosticCode) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == code)
    }

    fn one(
        location: LoweringLocation,
        code: LoweringDiagnosticCode,
        message: impl Into<String>,
    ) -> Self {
        Self {
            diagnostics: vec![LoweringDiagnostic {
                location,
                code,
                message: message.into(),
            }],
        }
    }

    fn verification(errors: VerificationErrors) -> Self {
        let diagnostics = errors
            .into_diagnostics()
            .into_iter()
            .map(|diagnostic| LoweringDiagnostic {
                location: LoweringLocation {
                    module: diagnostic.location.module,
                    kernel: diagnostic.location.kernel,
                    function: diagnostic.location.function,
                    block: diagnostic.location.block,
                    operation: diagnostic.location.operation,
                },
                code: LoweringDiagnosticCode::InputVerification(diagnostic.code),
                message: diagnostic.message,
            })
            .collect();
        Self { diagnostics }
    }
}

impl fmt::Display for LoweringErrors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "AMDGPU lowering failed with {} diagnostic(s)",
            self.diagnostics.len()
        )?;
        for diagnostic in &self.diagnostics {
            writeln!(formatter, "  {diagnostic}")?;
        }
        Ok(())
    }
}

impl Error for LoweringErrors {}

/// Lowers one exact kernel to deterministic textual AMDGPU LLVM IR.
///
/// This is a deliberately small, non-production lowering seam. It always calls
/// [`verify_module`] before selecting the kernel, then rejects every valid IR construct outside
/// the documented G1 subset. The launch extent remains a host contract; LLVM IR records the
/// required workgroup size, while launch-grid selection remains outside this API.
pub fn lower_kernel_to_llvm_ir(
    module: &Module,
    kernel_id: &KernelId,
) -> Result<String, LoweringErrors> {
    lower_kernel_to_llvm_ir_for_target(module, kernel_id, LoweringTarget::Baseline)
}

/// Lowers one kernel for the strict gfx942 floating-point profile.
///
/// This profile fixes the processor, preserves `f32` denormals, disables fast-math assumptions,
/// and admits only the explicit floating-point contracts represented by [`FloatOperation`].
/// OCML calls remain unresolved declarations and must be closed by an authenticated direct-LLVM
/// link plan before the module can become executable.
pub fn lower_kernel_to_gfx942_llvm_ir(
    module: &Module,
    kernel_id: &KernelId,
) -> Result<String, LoweringErrors> {
    lower_kernel_to_llvm_ir_for_target(module, kernel_id, LoweringTarget::Gfx942StrictFloatV1)
}

/// Lowers one kernel only when Kernel IR retains the exact gfx942:xnack- binding.
///
/// Unlike [`lower_kernel_to_gfx942_llvm_ir`], this API requires the target
/// identity on the module, kernel, and entry function. It emits the canonical
/// gfx942 data layout, explicit `-xnack`, exact wave-width features, and the
/// strict floating-point attribute set on the kernel definition.
pub fn lower_kernel_to_gfx942_xnack_minus_llvm_ir(
    module: &Module,
    kernel_id: &KernelId,
) -> Result<String, LoweringErrors> {
    lower_kernel_to_llvm_ir_for_target(module, kernel_id, LoweringTarget::Gfx942XnackMinusV1)
}

/// Lowers one kernel only when Kernel IR retains the exact gfx950:xnack- binding.
///
/// The returned module binds the reviewed AMDGPU data layout, gfx950 processor,
/// explicit `-xnack`, Wave64 policy, and strict floating-point attributes. It
/// remains inert LLVM text and grants no code-object or execution authority.
pub fn lower_kernel_to_gfx950_xnack_minus_llvm_ir(
    module: &Module,
    kernel_id: &KernelId,
) -> Result<String, LoweringErrors> {
    lower_kernel_to_llvm_ir_for_target(module, kernel_id, LoweringTarget::Gfx950XnackMinusV1)
}

fn lower_kernel_to_llvm_ir_for_target(
    module: &Module,
    kernel_id: &KernelId,
    target: LoweringTarget,
) -> Result<String, LoweringErrors> {
    verify_module(module).map_err(LoweringErrors::verification)?;

    let matches = module
        .kernels
        .iter()
        .filter(|kernel| &kernel.id == kernel_id)
        .collect::<Vec<_>>();
    let kernel = match matches.as_slice() {
        [] => {
            return Err(LoweringErrors::one(
                LoweringLocation::module(module),
                LoweringDiagnosticCode::MissingKernel,
                format!("kernel {kernel_id} is not in the module"),
            ));
        }
        [kernel] => *kernel,
        _ => {
            return Err(LoweringErrors::one(
                LoweringLocation::module(module),
                LoweringDiagnosticCode::AmbiguousKernel,
                format!("kernel identity {kernel_id} is ambiguous"),
            ));
        }
    };

    if !is_safe_symbol(kernel.id.as_str()) {
        return Err(LoweringErrors::one(
            LoweringLocation::kernel(module, kernel),
            LoweringDiagnosticCode::UnsafeSymbolName,
            "kernel identity is not a safe unquoted LLVM symbol",
        ));
    }
    if is_reserved_float_support_symbol(kernel.id.as_str()) {
        return Err(LoweringErrors::one(
            LoweringLocation::kernel(module, kernel),
            LoweringDiagnosticCode::ConflictingSymbol,
            "kernel identity collides with reserved gfx942 floating-point support",
        ));
    }

    let module_wave = validate_capabilities(
        LoweringLocation::module(module),
        &module.required_capabilities,
        "module",
        target,
    )?;
    let kernel_wave = validate_capabilities(
        LoweringLocation::kernel(module, kernel),
        &kernel.required_capabilities,
        "kernel",
        target,
    )?;

    let workgroup_size = validate_launch(module, kernel, target)?;
    let entry = module
        .functions
        .iter()
        .find(|function| function.id == kernel.entry)
        .expect("verify_module established the kernel entry");
    entry.body.as_ref().ok_or_else(|| {
        LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::KernelEntryDeclaration,
            "kernel entry must be a definition",
        )
    })?;

    let function_wave = validate_capabilities(
        LoweringLocation::function(module, kernel, entry),
        &entry.required_capabilities,
        "entry function",
        target,
    )?;
    if let Some(exact_target) = target.exact_target_binding() {
        require_exact_kernel_binding(module, kernel, entry, exact_target)?;
    }
    validate_matrix_frontend_abi_binding(module, kernel, entry, target)?;
    let wave_widths = [module_wave, kernel_wave, function_wave]
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>();
    if wave_widths.len() > 1 {
        return Err(LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::UnsupportedCapability,
            format!("conflicting exact wave-width requirements: {wave_widths:?}"),
        ));
    }
    let wave_width = wave_widths.first().copied();
    if !entry.signature.results.is_empty() {
        return Err(LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::UnsupportedResults,
            "G1 kernel entries must return void",
        ));
    }

    let mut lowerer =
        FunctionLowerer::new(module, kernel, entry, workgroup_size, wave_width, target);
    preflight_function(&mut lowerer)?;
    lowerer.emit()
}

/// Lowers one verified kernel-IR module to one deterministic textual AMDGPU LLVM module.
///
/// This is an inert compiler-module construction primitive. It performs no linking, target
/// selection, optimization, publication, or code-object generation. Kernel entries are emitted
/// in kernel-identity order. Non-kernel definitions and external declarations are emitted once in
/// function-identity order, while each function body preserves its verified block and operation
/// order. All functions are preflighted before any output is returned.
///
/// The current bounded feature slice supports void or single-result scalar/pointer helper ABIs.
/// Slice ABIs remain kernel-entry-only. Calls to kernel entry functions and context-dependent
/// operations in helpers are rejected. Helper wave modes are resolved through a bounded SCC call
/// graph before lowering, and textual output is capacity-limited and returned atomically.
///
/// The text binds the AMDGPU target triple only. Target data layout, processor identity, and code
/// object version are deliberately absent and remain blockers for artifact construction.
pub fn lower_compiler_module_to_llvm_ir(module: &Module) -> Result<String, LoweringErrors> {
    lower_compiler_module_to_llvm_ir_for_target(module, LoweringTarget::Baseline, None, true)
}

/// Lowers a complete compiler module for the strict gfx942 floating-point profile.
pub fn lower_compiler_module_to_gfx942_llvm_ir(module: &Module) -> Result<String, LoweringErrors> {
    lower_compiler_module_to_llvm_ir_for_target(
        module,
        LoweringTarget::Gfx942StrictFloatV1,
        None,
        true,
    )
}

/// Lowers a complete compiler module for the exact gfx950:xnack- profile.
pub fn lower_compiler_module_to_gfx950_xnack_minus_llvm_ir(
    module: &Module,
) -> Result<String, LoweringErrors> {
    lower_compiler_module_to_llvm_ir_for_target(
        module,
        LoweringTarget::Gfx950XnackMinusV1,
        None,
        true,
    )
}

/// Lowers a verified helper-only module for the strict gfx942 profile.
///
/// Every defined function must carry an exact wave-width capability. This API
/// emits ordinary AMDGPU device functions and declarations, never kernel
/// entries, launch metadata, a code object, or execution authority.
pub fn lower_device_module_to_gfx942_llvm_ir(module: &Module) -> Result<String, LoweringErrors> {
    lower_compiler_module_to_llvm_ir_for_target(
        module,
        LoweringTarget::Gfx942StrictFloatV1,
        None,
        false,
    )
}

/// Lowers a verified helper-only module for the exact gfx942:xnack- profile.
///
/// In addition to the strict gfx942 processor and floating-point attributes,
/// this entry point emits the reviewed target data layout and binds `-xnack`
/// in every defined function's target-feature set.
pub fn lower_device_module_to_gfx942_xnack_minus_llvm_ir(
    module: &Module,
) -> Result<String, LoweringErrors> {
    lower_compiler_module_to_llvm_ir_for_target(
        module,
        LoweringTarget::Gfx942XnackMinusV1,
        None,
        false,
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gfx942KernelLaunchPolicyV1 {
    kernel_id: KernelId,
    launch_bounds: Gfx942LaunchBoundsV1,
}

impl Gfx942KernelLaunchPolicyV1 {
    pub fn new(kernel_id: KernelId, launch_bounds: Gfx942LaunchBoundsV1) -> Self {
        Self {
            kernel_id,
            launch_bounds,
        }
    }

    pub const fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    pub const fn launch_bounds(&self) -> Gfx942LaunchBoundsV1 {
        self.launch_bounds
    }
}

/// Emits exact AMD flat-workgroup and waves-per-EU attributes for every kernel.
///
/// This does not translate CUDA's minimum-blocks-per-SM semantics.
pub fn lower_compiler_module_to_gfx942_llvm_ir_with_launch_policies(
    module: &Module,
    launch_policies: &[Gfx942KernelLaunchPolicyV1],
) -> Result<String, LoweringErrors> {
    lower_compiler_module_to_llvm_ir_for_target(
        module,
        LoweringTarget::Gfx942StrictFloatV1,
        Some(launch_policies),
        true,
    )
}

fn lower_compiler_module_to_llvm_ir_for_target(
    module: &Module,
    target: LoweringTarget,
    launch_policies: Option<&[Gfx942KernelLaunchPolicyV1]>,
    require_kernel: bool,
) -> Result<String, LoweringErrors> {
    if require_kernel && module.kernels.is_empty() {
        return Err(LoweringErrors::one(
            LoweringLocation::module(module),
            LoweringDiagnosticCode::MissingKernel,
            "compiler-module lowering requires at least one kernel entry",
        ));
    }
    verify_module(module).map_err(LoweringErrors::verification)?;

    if let Some(exact_target) = target.exact_target_binding() {
        for kernel in &module.kernels {
            let entry = module
                .functions
                .iter()
                .find(|function| function.id == kernel.entry)
                .expect("verify_module established every kernel entry");
            require_exact_kernel_binding(module, kernel, entry, exact_target)?;
        }
    }

    let module_wave = validate_capabilities(
        LoweringLocation::module(module),
        &module.required_capabilities,
        "module",
        target,
    )?;
    let mut kernels = module.kernels.iter().collect::<Vec<_>>();
    kernels.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
    let mut functions = module.functions.iter().collect::<Vec<_>>();
    functions.sort_by(|lhs, rhs| lhs.id.cmp(&rhs.id));
    for kernel in &kernels {
        let entry = module
            .function(&kernel.entry)
            .expect("verify_module established the kernel entry");
        validate_matrix_frontend_abi_binding(module, kernel, entry, target)?;
    }

    let launch_policy_map = validate_launch_policies(module, &kernels, launch_policies)?;

    let mut entries = BTreeMap::<FunctionId, &Kernel>::new();
    let mut emitted_symbols = BTreeMap::<String, String>::new();
    for kernel in &kernels {
        if !is_safe_symbol(kernel.id.as_str()) {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::UnsafeSymbolName,
                "kernel identity is not a safe unquoted LLVM symbol",
            ));
        }
        if is_reserved_float_support_symbol(kernel.id.as_str()) {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::ConflictingSymbol,
                "kernel identity collides with reserved gfx942 floating-point support",
            ));
        }
        if let Some(previous) = entries.insert(kernel.entry.clone(), kernel) {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::ConflictingSymbol,
                format!(
                    "kernel entry function {} is already emitted as kernel {}; one definition cannot back multiple exported entries",
                    kernel.entry, previous.id
                ),
            ));
        }
        reserve_emitted_symbol(
            &mut emitted_symbols,
            kernel.id.as_str(),
            format!("kernel {}", kernel.id),
            LoweringLocation::kernel(module, kernel),
        )?;
    }
    for kernel in &kernels {
        let entry = module
            .function(&kernel.entry)
            .expect("verify_module established the kernel entry");
        let body = entry.body.as_ref().expect("verified kernel entry body");
        for block in &body.blocks {
            for (operation_index, operation) in block.operations.iter().enumerate() {
                if !matches!(&operation.kind, OperationKind::WorkgroupMemory(_)) {
                    continue;
                }
                let result = operation
                    .results
                    .first()
                    .expect("verify_module established the LDS result");
                let emitted = lds_symbol(kernel, result.id);
                reserve_emitted_symbol(
                    &mut emitted_symbols,
                    emitted.strip_prefix('@').expect("LDS symbols start with @"),
                    format!("kernel {} LDS value {}", kernel.id, result.id),
                    LoweringLocation::operation(module, kernel, entry, block.id, operation_index),
                )?;
            }
        }
    }

    let mut call_symbols = BTreeMap::<FunctionId, String>::new();
    let mut declarations = Vec::new();
    let mut helper_definitions = Vec::new();
    for function in &functions {
        if entries.contains_key(&function.id) {
            continue;
        }
        if FloatOperation::from_intrinsic_id(&function.id).is_some() {
            continue;
        }
        if AmdGpuDiagnosticOperation::from_intrinsic_id(&function.id).is_some() {
            continue;
        }
        let location = LoweringLocation::device_function(module, function);
        if !is_safe_symbol(function.id.as_str()) {
            return Err(LoweringErrors::one(
                location,
                LoweringDiagnosticCode::UnsafeSymbolName,
                "device function identity is not a safe unquoted LLVM symbol",
            ));
        }
        if is_reserved_float_support_symbol(function.id.as_str()) {
            return Err(LoweringErrors::one(
                location,
                LoweringDiagnosticCode::ConflictingSymbol,
                "device function identity collides with reserved gfx942 floating-point support",
            ));
        }
        reserve_emitted_symbol(
            &mut emitted_symbols,
            function.id.as_str(),
            format!("device function {}", function.id),
            location.clone(),
        )?;
        validate_device_signature(module, function, target)?;
        call_symbols.insert(function.id.clone(), function.id.as_str().to_string());
        match function.role {
            FunctionRole::InternalHelper | FunctionRole::DeviceFfiExport => {
                helper_definitions.push(*function);
            }
            FunctionRole::ExternalImport => {
                if function.required_capabilities.iter().any(|capability| {
                    !matches!(
                        capability,
                        TargetCapability::Float16 | TargetCapability::BFloat16
                    )
                }) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedCapability,
                        "external declarations may carry only narrow-float ABI capabilities in the gfx942 textual compiler-module slice",
                    ));
                }
                validate_capabilities(
                    location,
                    &function.required_capabilities,
                    "external declaration",
                    target,
                )?;
                declarations.push(*function);
            }
            FunctionRole::KernelEntry => {
                unreachable!("verify_module rejects unreferenced KernelEntry definitions")
            }
        }
    }

    let wave_plan =
        infer_effective_wave_widths(module, module_wave, &kernels, &helper_definitions, target)?;

    let mut kernel_lowerers = Vec::with_capacity(kernels.len());
    for kernel in &kernels {
        let workgroup_size = validate_launch(module, kernel, target)?;
        let launch_bounds = launch_policy_map
            .as_ref()
            .map(|policies| policies[&kernel.id]);
        let flat_workgroup_size = checked_flat_workgroup_size(workgroup_size)
            .expect("validate_launch established a bounded flat workgroup size");
        if launch_bounds
            .is_some_and(|bounds| !bounds.admits_flat_workgroup_size(flat_workgroup_size))
        {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::InvalidLaunchPolicy,
                format!(
                    "exact workgroup size {} is outside the admitted flat workgroup range",
                    flat_workgroup_size
                ),
            ));
        }
        let entry = module
            .function(&kernel.entry)
            .expect("verify_module established the kernel entry");
        let wave_width = wave_plan.kernels[&kernel.id];
        let mut lowerer = FunctionLowerer::compiler_module_kernel(
            module,
            kernel,
            entry,
            workgroup_size,
            wave_width,
            &call_symbols,
            target,
            launch_bounds,
        );
        preflight_function(&mut lowerer)?;
        kernel_lowerers.push(lowerer);
    }

    let mut helper_lowerers = Vec::with_capacity(helper_definitions.len());
    for function in helper_definitions {
        let wave_width = wave_plan.helpers[&function.id];
        let mut lowerer = FunctionLowerer::compiler_module_device_function(
            module,
            function,
            wave_width,
            &call_symbols,
            target,
        );
        preflight_function(&mut lowerer)?;
        helper_lowerers.push(lowerer);
    }

    emit_compiler_module(
        module,
        &kernel_lowerers,
        &helper_lowerers,
        &declarations,
        target,
    )
}

fn validate_launch_policies(
    module: &Module,
    kernels: &[&Kernel],
    policies: Option<&[Gfx942KernelLaunchPolicyV1]>,
) -> Result<Option<BTreeMap<KernelId, Gfx942LaunchBoundsV1>>, LoweringErrors> {
    let Some(policies) = policies else {
        return Ok(None);
    };
    if policies.len() != kernels.len() {
        return Err(LoweringErrors::one(
            LoweringLocation::module(module),
            LoweringDiagnosticCode::InvalidLaunchPolicy,
            format!(
                "strict launch-policy lowering requires exactly one policy per kernel: expected {}, found {}",
                kernels.len(),
                policies.len()
            ),
        ));
    }
    let kernel_ids = kernels
        .iter()
        .map(|kernel| &kernel.id)
        .collect::<BTreeSet<_>>();
    let mut result = BTreeMap::new();
    for policy in policies {
        if !kernel_ids.contains(policy.kernel_id()) {
            return Err(LoweringErrors::one(
                LoweringLocation::module(module),
                LoweringDiagnosticCode::InvalidLaunchPolicy,
                format!("launch policy names unknown kernel {}", policy.kernel_id()),
            ));
        }
        if result
            .insert(policy.kernel_id.clone(), policy.launch_bounds)
            .is_some()
        {
            return Err(LoweringErrors::one(
                LoweringLocation::module(module),
                LoweringDiagnosticCode::InvalidLaunchPolicy,
                format!(
                    "launch policy for kernel {} is duplicated",
                    policy.kernel_id()
                ),
            ));
        }
    }
    Ok(Some(result))
}

fn reserve_emitted_symbol(
    symbols: &mut BTreeMap<String, String>,
    symbol: &str,
    owner: String,
    location: LoweringLocation,
) -> Result<(), LoweringErrors> {
    if let Some(previous) = symbols.insert(symbol.to_string(), owner.clone()) {
        return Err(LoweringErrors::one(
            location,
            LoweringDiagnosticCode::ConflictingSymbol,
            format!("LLVM symbol {symbol:?} is claimed by both {previous} and {owner}"),
        ));
    }
    Ok(())
}

fn validate_device_signature(
    module: &Module,
    function: &Function,
    target: LoweringTarget,
) -> Result<(), LoweringErrors> {
    let location = LoweringLocation::device_function(module, function);
    for (index, ty) in function.signature.parameters.iter().enumerate() {
        validate_device_abi_type(ty, &location, target).map_err(|error| {
            LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedParameter,
                format!("unsupported device parameter {index}: {error}"),
            )
        })?;
    }
    if function.signature.results.len() > 1 {
        return Err(LoweringErrors::one(
            location,
            LoweringDiagnosticCode::UnsupportedResults,
            "device functions may return at most one scalar or pointer value",
        ));
    }
    if let Some(result) = function.signature.results.first() {
        validate_device_abi_type(result, &location, target).map_err(|error| {
            LoweringErrors::one(
                location,
                LoweringDiagnosticCode::UnsupportedResults,
                format!("unsupported device result: {error}"),
            )
        })?;
    }
    Ok(())
}

fn validate_device_abi_type(
    ty: &Type,
    location: &LoweringLocation,
    target: LoweringTarget,
) -> Result<(), String> {
    match ty {
        Type::Scalar(scalar) if supported_scalar(*scalar, target) => Ok(()),
        Type::Pointer(_) => {
            validate_device_pointer(ty, location, target).map_err(|error| error.to_string())
        }
        _ => Err(format!("{ty:?}")),
    }
}

fn unique_wave_width(
    location: LoweringLocation,
    widths: [Option<WaveWidth>; 3],
) -> Result<Option<WaveWidth>, LoweringErrors> {
    let widths = widths.into_iter().flatten().collect::<BTreeSet<_>>();
    if widths.len() > 1 {
        return Err(LoweringErrors::one(
            location,
            LoweringDiagnosticCode::UnsupportedCapability,
            format!("conflicting exact wave-width requirements: {widths:?}"),
        ));
    }
    Ok(widths.first().copied())
}

struct EffectiveWavePlan {
    kernels: BTreeMap<KernelId, Option<WaveWidth>>,
    helpers: BTreeMap<FunctionId, Option<WaveWidth>>,
}

fn infer_effective_wave_widths(
    module: &Module,
    module_wave: Option<WaveWidth>,
    kernels: &[&Kernel],
    helpers: &[&Function],
    target: LoweringTarget,
) -> Result<EffectiveWavePlan, LoweringErrors> {
    enforce_call_graph_limit(
        module,
        "compiler-module graph functions",
        module.functions.len(),
        MAX_COMPILER_MODULE_GRAPH_FUNCTIONS,
    )?;
    enforce_call_graph_limit(
        module,
        "compiler-module graph kernels",
        kernels.len(),
        MAX_COMPILER_MODULE_GRAPH_KERNELS,
    )?;

    let helper_indices = helpers
        .iter()
        .enumerate()
        .map(|(index, function)| (function.id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut edge_count = 0usize;
    let mut adjacency = Vec::with_capacity(helpers.len());
    for function in helpers {
        adjacency.push(helper_callees(
            module,
            function,
            &helper_indices,
            &mut edge_count,
        )?);
    }

    let (component_of, components) = strongly_connected_components(&adjacency);
    let mut component_edges = vec![BTreeSet::new(); components.len()];
    for (caller, callees) in adjacency.iter().enumerate() {
        for callee in callees {
            let source = component_of[caller];
            let target = component_of[*callee];
            if source != target {
                component_edges[source].insert(target);
            }
        }
    }

    let mut component_claims = vec![BTreeSet::new(); components.len()];
    for (index, function) in helpers.iter().enumerate() {
        let function_wave = validate_capabilities(
            LoweringLocation::device_function(module, function),
            &function.required_capabilities,
            "device function",
            target,
        )?;
        component_claims[component_of[index]]
            .extend([module_wave, function_wave].into_iter().flatten());
    }
    for (component, claims) in component_claims.iter().enumerate() {
        reject_mixed_component_modes(module, helpers, &components[component], claims)?;
    }

    let mut assignments = vec![BTreeSet::new(); components.len()];
    let mut kernel_reachable = vec![false; components.len()];
    let mut kernel_modes = BTreeMap::new();
    for kernel in kernels {
        let entry = module
            .function(&kernel.entry)
            .expect("verified kernel entry");
        let kernel_wave = validate_capabilities(
            LoweringLocation::kernel(module, kernel),
            &kernel.required_capabilities,
            "kernel",
            target,
        )?;
        let entry_wave = validate_capabilities(
            LoweringLocation::function(module, kernel, entry),
            &entry.required_capabilities,
            "entry function",
            target,
        )?;
        let root_wave = unique_wave_width(
            LoweringLocation::function(module, kernel, entry),
            [module_wave, kernel_wave, entry_wave],
        )?;
        let direct = helper_callees(module, entry, &helper_indices, &mut edge_count)?
            .into_iter()
            .map(|helper| component_of[helper]);
        let reachable = reachable_components(direct, &component_edges);
        let mut modes = root_wave.into_iter().collect::<BTreeSet<_>>();
        for component in &reachable {
            kernel_reachable[*component] = true;
            modes.extend(component_claims[*component].iter().copied());
        }
        if modes.len() > 1 {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::IncompatibleWaveCallGraph,
                format!(
                    "kernel {} reaches helper SCCs with incompatible exact wave modes {modes:?}",
                    kernel.id
                ),
            ));
        }
        if !reachable.is_empty() && modes.is_empty() {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::MissingWaveWidth,
                format!(
                    "kernel {} reaches helpers but neither the kernel closure nor module declares an exact wave mode",
                    kernel.id
                ),
            ));
        }
        let mode = modes.first().copied();
        if let Some(mode) = mode {
            for component in reachable {
                assignments[component].insert(mode);
            }
        }
        kernel_modes.insert(kernel.id.clone(), mode);
    }

    for (component, modes) in assignments.iter().enumerate() {
        if modes.len() > 1 {
            return Err(mixed_assignment_error(
                module,
                helpers,
                &components[component],
                modes,
            ));
        }
    }

    let mut unreachable_indegree = vec![0usize; components.len()];
    for (source, targets) in component_edges.iter().enumerate() {
        if kernel_reachable[source] {
            continue;
        }
        for target in targets {
            if !kernel_reachable[*target] {
                unreachable_indegree[*target] += 1;
            }
        }
    }
    let mut non_kernel_roots = (0..components.len())
        .filter(|component| {
            !kernel_reachable[*component]
                && (unreachable_indegree[*component] == 0
                    || components[*component]
                        .iter()
                        .any(|helper| helpers[*helper].role == FunctionRole::DeviceFfiExport))
        })
        .collect::<BTreeSet<_>>();
    while let Some(root) = non_kernel_roots.pop_first() {
        if component_claims[root].is_empty() {
            let function = helpers[components[root][0]];
            return Err(LoweringErrors::one(
                LoweringLocation::device_function(module, function),
                LoweringDiagnosticCode::MissingWaveWidth,
                format!(
                    "non-kernel-reachable helper SCC [{}] requires an explicit exact wave mode",
                    component_names(helpers, &components[root])
                ),
            ));
        }
        let reachable = reachable_components([root], &component_edges);
        let mut modes = BTreeSet::new();
        for component in &reachable {
            modes.extend(component_claims[*component].iter().copied());
            modes.extend(assignments[*component].iter().copied());
        }
        if modes.len() > 1 {
            return Err(mixed_assignment_error(
                module,
                helpers,
                &components[root],
                &modes,
            ));
        }
        let mode = *modes.first().expect("root has an explicit wave mode");
        for component in reachable {
            assignments[component].insert(mode);
        }
    }

    let mut helper_modes = BTreeMap::new();
    for (index, function) in helpers.iter().enumerate() {
        let component = component_of[index];
        let mut modes = component_claims[component].clone();
        modes.extend(assignments[component].iter().copied());
        if modes.len() > 1 {
            return Err(mixed_assignment_error(
                module,
                helpers,
                &components[component],
                &modes,
            ));
        }
        let mode = modes.first().copied().ok_or_else(|| {
            LoweringErrors::one(
                LoweringLocation::device_function(module, function),
                LoweringDiagnosticCode::MissingWaveWidth,
                "helper has no effective exact wave mode after bounded call-graph propagation",
            )
        })?;
        helper_modes.insert(function.id.clone(), Some(mode));
    }

    Ok(EffectiveWavePlan {
        kernels: kernel_modes,
        helpers: helper_modes,
    })
}

fn helper_callees(
    module: &Module,
    function: &Function,
    helper_indices: &BTreeMap<FunctionId, usize>,
    edge_count: &mut usize,
) -> Result<Vec<usize>, LoweringErrors> {
    let mut callees = BTreeSet::new();
    let body = function.body.as_ref().expect("definition required");
    for operation in body.blocks.iter().flat_map(|block| &block.operations) {
        let OperationKind::Call { callee, .. } = &operation.kind else {
            continue;
        };
        *edge_count = edge_count.saturating_add(1);
        enforce_call_graph_limit(
            module,
            "compiler-module call edges",
            *edge_count,
            MAX_COMPILER_MODULE_CALL_EDGES,
        )?;
        if let Some(index) = helper_indices.get(callee) {
            callees.insert(*index);
        }
    }
    Ok(callees.into_iter().collect())
}

fn enforce_call_graph_limit(
    module: &Module,
    field: &'static str,
    actual: usize,
    max: usize,
) -> Result<(), LoweringErrors> {
    if actual <= max {
        return Ok(());
    }
    Err(LoweringErrors::one(
        LoweringLocation::module(module),
        LoweringDiagnosticCode::ResourceLimit,
        format!("{field} count {actual} exceeds limit {max}"),
    ))
}

fn strongly_connected_components(adjacency: &[Vec<usize>]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut reverse = vec![Vec::new(); adjacency.len()];
    for (source, targets) in adjacency.iter().enumerate() {
        for target in targets {
            reverse[*target].push(source);
        }
    }
    for targets in &mut reverse {
        targets.sort_unstable();
    }

    let mut visited = vec![false; adjacency.len()];
    let mut finished = Vec::with_capacity(adjacency.len());
    for start in 0..adjacency.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next)) = stack.last_mut() {
            if *next < adjacency[*node].len() {
                let target = adjacency[*node][*next];
                *next += 1;
                if !visited[target] {
                    visited[target] = true;
                    stack.push((target, 0));
                }
            } else {
                finished.push(*node);
                stack.pop();
            }
        }
    }

    let mut component_of = vec![usize::MAX; adjacency.len()];
    let mut components = Vec::new();
    for start in finished.into_iter().rev() {
        if component_of[start] != usize::MAX {
            continue;
        }
        let component = components.len();
        let mut members = Vec::new();
        let mut stack = vec![start];
        component_of[start] = component;
        while let Some(node) = stack.pop() {
            members.push(node);
            for predecessor in reverse[node].iter().rev() {
                if component_of[*predecessor] == usize::MAX {
                    component_of[*predecessor] = component;
                    stack.push(*predecessor);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }
    (component_of, components)
}

fn reachable_components(
    roots: impl IntoIterator<Item = usize>,
    edges: &[BTreeSet<usize>],
) -> BTreeSet<usize> {
    let mut reachable = BTreeSet::new();
    let mut pending = roots.into_iter().collect::<Vec<_>>();
    while let Some(component) = pending.pop() {
        if !reachable.insert(component) {
            continue;
        }
        pending.extend(edges[component].iter().rev().copied());
    }
    reachable
}

fn reject_mixed_component_modes(
    module: &Module,
    helpers: &[&Function],
    component: &[usize],
    modes: &BTreeSet<WaveWidth>,
) -> Result<(), LoweringErrors> {
    if modes.len() <= 1 {
        return Ok(());
    }
    Err(mixed_assignment_error(module, helpers, component, modes))
}

fn mixed_assignment_error(
    module: &Module,
    helpers: &[&Function],
    component: &[usize],
    modes: &BTreeSet<WaveWidth>,
) -> LoweringErrors {
    let function = helpers[component[0]];
    LoweringErrors::one(
        LoweringLocation::device_function(module, function),
        LoweringDiagnosticCode::IncompatibleWaveCallGraph,
        format!(
            "helper SCC [{}] is reachable with incompatible exact wave modes {modes:?}",
            component_names(helpers, component)
        ),
    )
}

fn component_names(helpers: &[&Function], component: &[usize]) -> String {
    component
        .iter()
        .map(|index| helpers[*index].id.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn preflight_function(lowerer: &mut FunctionLowerer<'_>) -> Result<(), LoweringErrors> {
    validate_reducible_cfg(lowerer)?;
    validate_convergent_cfg(lowerer)?;
    lowerer.validate_parameters()?;
    let body = lowerer.function.body.as_ref().expect("definition required");
    for block in &body.blocks {
        lowerer.validate_block(block)?;
    }
    lowerer.validate_block_arguments()?;
    lowerer.validate_lds_addressability()
}

fn validate_reducible_cfg(lowerer: &FunctionLowerer<'_>) -> Result<(), LoweringErrors> {
    let cyclic = lowerer.control_flow.irreducible_blocks();
    if cyclic.is_empty() {
        return Ok(());
    }
    let labels = cyclic
        .iter()
        .map(|block| block_label(*block))
        .collect::<Vec<_>>();
    Err(LoweringErrors::one(
        lowerer.block_location(cyclic[0]),
        LoweringDiagnosticCode::IrreducibleControlFlow,
        format!(
            "control flow remains cyclic after removing dominance backedges: {}",
            labels.join(", ")
        ),
    ))
}

fn validate_convergent_cfg(lowerer: &FunctionLowerer<'_>) -> Result<(), LoweringErrors> {
    let body = lowerer.function.body.as_ref().expect("definition required");
    let convergent_operations = body
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(|(operation, value)| {
                    matches!(
                        value.kind,
                        OperationKind::WorkgroupBarrier(_)
                            | OperationKind::Wave(_)
                            | OperationKind::Matrix(_)
                            | OperationKind::Gfx950LdsTranspose(_)
                    )
                    .then_some((block.id, operation))
                })
        })
        .collect::<Vec<_>>();
    if convergent_operations.is_empty() {
        return Ok(());
    }

    let report = fe2o3_kernel_analysis::analyze_kernel_entry(lowerer.module, lowerer.function);
    if let Some(diagnostic) = report.diagnostics().iter().find(|diagnostic| {
        matches!(
            diagnostic,
            fe2o3_kernel_analysis::Diagnostic::Unsupported { .. }
        )
    }) {
        return Err(LoweringErrors::one(
            lowerer.function_location(),
            LoweringDiagnosticCode::UnprovenBarrierConvergence,
            format!(
                "target-neutral uniformity analysis could not prove convergence: {diagnostic:?}"
            ),
        ));
    }
    for (block, operation) in convergent_operations {
        let operation_value = &lowerer.block(block).operations[operation];
        let scope = match &operation_value.kind {
            OperationKind::WorkgroupBarrier(barrier) => barrier.convergence.scope(),
            OperationKind::Wave(wave) => wave.convergence.scope(),
            OperationKind::Matrix(matrix) => matrix.convergence.scope(),
            OperationKind::Gfx950LdsTranspose(transpose) => transpose.convergence.scope(),
            _ => unreachable!("convergent operation inventory is exact"),
        };
        let control = report.block_control(block);
        if !control.is_uniform_for(scope) {
            return Err(LoweringErrors::one(
                lowerer.operation_location(block, operation),
                LoweringDiagnosticCode::UnprovenBarrierConvergence,
                format!(
                    "convergent operation requires {scope:?} uniform control, but analysis found {control:?}",
                ),
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct DiagnosticRequirements {
    clock: bool,
    trap: bool,
    debugtrap: bool,
}

impl DiagnosticRequirements {
    fn collect<'a>(lowerers: impl Iterator<Item = &'a FunctionLowerer<'a>>) -> Self {
        let mut requirements = Self::default();
        for lowerer in lowerers {
            let body = lowerer.function.body.as_ref().expect("definition required");
            for operation in body.blocks.iter().flat_map(|block| &block.operations) {
                let OperationKind::Call { callee, arguments } = &operation.kind else {
                    continue;
                };
                let Some(diagnostic) =
                    AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
                else {
                    continue;
                };
                match diagnostic {
                    AmdGpuDiagnosticOperation::Clock32 => requirements.clock = true,
                    AmdGpuDiagnosticOperation::Trap
                    | AmdGpuDiagnosticOperation::AssertFail { .. } => requirements.trap = true,
                    AmdGpuDiagnosticOperation::DebugTrap => requirements.debugtrap = true,
                    AmdGpuDiagnosticOperation::ProfilingMarker { .. }
                    | AmdGpuDiagnosticOperation::Print { .. } => {}
                }
            }
        }
        requirements
    }

    const fn is_empty(&self) -> bool {
        !self.clock && !self.trap && !self.debugtrap
    }
}

fn emit_diagnostic_declarations(
    output: &mut dyn fmt::Write,
    requirements: &DiagnosticRequirements,
) {
    if requirements.clock {
        writeln!(output, "declare i64 @llvm.amdgcn.s.memrealtime()").unwrap();
    }
    if requirements.trap {
        writeln!(output, "declare void @llvm.trap()").unwrap();
    }
    if requirements.debugtrap {
        writeln!(output, "declare void @llvm.debugtrap()").unwrap();
    }
}

#[derive(Default)]
struct FloatRequirements {
    conversions: BTreeSet<FloatConversionKind>,
    widened_binary: BTreeSet<WidenedFloatBinaryOp>,
    math: BTreeSet<F32MathFunction>,
    packed_bf16_fma: bool,
}

impl FloatRequirements {
    fn collect<'a>(lowerers: impl Iterator<Item = &'a FunctionLowerer<'a>>) -> Self {
        let mut requirements = Self::default();
        for lowerer in lowerers {
            let body = lowerer.function.body.as_ref().expect("definition required");
            for operation in body.blocks.iter().flat_map(|block| &block.operations) {
                let OperationKind::Call { callee, arguments } = &operation.kind else {
                    continue;
                };
                let Some(float) = FloatOperation::from_intrinsic_call(callee, arguments) else {
                    continue;
                };
                match &float {
                    FloatOperation::Convert { kind, .. } => {
                        requirements.conversions.insert(*kind);
                    }
                    FloatOperation::WidenedBinary { format, op, .. } => {
                        requirements.widened_binary.insert(*op);
                        match format {
                            NarrowFloatFormat::F16 => {
                                requirements.conversions.extend([
                                    FloatConversionKind::F16ToF32,
                                    FloatConversionKind::F32ToF16RoundTiesEven,
                                ]);
                            }
                            NarrowFloatFormat::Bf16 => {
                                requirements.conversions.extend([
                                    FloatConversionKind::Bf16ToF32,
                                    FloatConversionKind::F32ToBf16RoundTiesEven,
                                ]);
                            }
                        }
                    }
                    FloatOperation::F32Math { function, .. } => {
                        requirements.math.insert(*function);
                    }
                    FloatOperation::Bf16x2FusedMultiplyAdd { .. } => {
                        requirements.packed_bf16_fma = true;
                        requirements.conversions.extend([
                            FloatConversionKind::Bf16ToF32,
                            FloatConversionKind::F32ToBf16RoundTiesEven,
                        ]);
                        requirements.math.insert(F32MathFunction::FusedMultiplyAdd);
                    }
                }
            }
        }
        requirements
    }

    fn is_empty(&self) -> bool {
        self.conversions.is_empty()
            && self.widened_binary.is_empty()
            && self.math.is_empty()
            && !self.packed_bf16_fma
    }
}

fn emit_float_support_declarations(
    output: &mut dyn fmt::Write,
    requirements: &FloatRequirements,
    target: LoweringTarget,
) {
    if requirements
        .conversions
        .contains(&FloatConversionKind::F16ToF32)
    {
        writeln!(output, "declare i32 @llvm.ctlz.i32(i32, i1 immarg)").unwrap();
    }
    for op in &requirements.widened_binary {
        writeln!(
            output,
            "declare float @{}(float, float, metadata, metadata)",
            constrained_binary_name(*op)
        )
        .unwrap();
    }
    for function in &requirements.math {
        match function.required_implementation() {
            F32MathImplementation::IeeeSqrtRoundTiesEvenIgnoreExceptionsV1 => {
                let arguments = if target == LoweringTarget::Gfx950XnackMinusV1 {
                    "float"
                } else {
                    "float, metadata, metadata"
                };
                writeln!(
                    output,
                    "declare float @{}({arguments})",
                    constrained_math_name(*function, target)
                )
                .unwrap();
            }
            F32MathImplementation::ConstrainedLlvm => {
                let arguments = match function {
                    F32MathFunction::FusedMultiplyAdd => "float, float, float, metadata, metadata",
                    F32MathFunction::Floor
                    | F32MathFunction::Ceil
                    | F32MathFunction::Truncate
                    | F32MathFunction::RoundTiesEven => "float, metadata",
                    _ => unreachable!("validated constrained function"),
                };
                writeln!(
                    output,
                    "declare float @{}({arguments})",
                    constrained_math_name(*function, target)
                )
                .unwrap();
            }
            F32MathImplementation::OcmlAbiV1 => {
                writeln!(output, "declare float @{}(float)", ocml_name(*function)).unwrap();
            }
        }
    }
}

fn emit_float_support_definitions(
    output: &mut dyn fmt::Write,
    requirements: &FloatRequirements,
    target: LoweringTarget,
) {
    let attributes = target.llvm_function_attributes();
    if requirements
        .conversions
        .contains(&FloatConversionKind::F16ToF32)
    {
        writeln!(
            output,
            "define internal float @__fe2o3_f16_to_f32_v1(i16 %bits) alwaysinline nounwind{attributes} {{"
        )
        .unwrap();
        writeln!(output, "entry:").unwrap();
        writeln!(output, "  %wide = zext i16 %bits to i32").unwrap();
        writeln!(output, "  %sign16 = and i32 %wide, 32768").unwrap();
        writeln!(output, "  %sign = shl i32 %sign16, 16").unwrap();
        writeln!(output, "  %exponent.shift = lshr i32 %wide, 10").unwrap();
        writeln!(output, "  %exponent = and i32 %exponent.shift, 31").unwrap();
        writeln!(output, "  %fraction = and i32 %wide, 1023").unwrap();
        writeln!(output, "  %fraction.f32 = shl i32 %fraction, 13").unwrap();
        writeln!(output, "  %normal.exponent.add = add i32 %exponent, 112").unwrap();
        writeln!(
            output,
            "  %normal.exponent = shl i32 %normal.exponent.add, 23"
        )
        .unwrap();
        writeln!(
            output,
            "  %normal.payload = or i32 %normal.exponent, %fraction.f32"
        )
        .unwrap();
        writeln!(output, "  %normal = or i32 %sign, %normal.payload").unwrap();
        writeln!(
            output,
            "  %special.payload = or i32 2139095040, %fraction.f32"
        )
        .unwrap();
        writeln!(output, "  %special = or i32 %sign, %special.payload").unwrap();
        writeln!(
            output,
            "  %leading = call i32 @llvm.ctlz.i32(i32 %fraction, i1 false)"
        )
        .unwrap();
        writeln!(output, "  %subnormal.shift = sub i32 %leading, 21").unwrap();
        writeln!(
            output,
            "  %subnormal.normalized = shl i32 %fraction, %subnormal.shift"
        )
        .unwrap();
        writeln!(
            output,
            "  %subnormal.fraction16 = and i32 %subnormal.normalized, 1023"
        )
        .unwrap();
        writeln!(
            output,
            "  %subnormal.fraction = shl i32 %subnormal.fraction16, 13"
        )
        .unwrap();
        writeln!(
            output,
            "  %subnormal.exponent.raw = sub i32 113, %subnormal.shift"
        )
        .unwrap();
        writeln!(
            output,
            "  %subnormal.exponent = shl i32 %subnormal.exponent.raw, 23"
        )
        .unwrap();
        writeln!(
            output,
            "  %subnormal.payload = or i32 %subnormal.exponent, %subnormal.fraction"
        )
        .unwrap();
        writeln!(output, "  %subnormal = or i32 %sign, %subnormal.payload").unwrap();
        writeln!(output, "  %fraction.zero = icmp eq i32 %fraction, 0").unwrap();
        writeln!(
            output,
            "  %zero.or.subnormal = select i1 %fraction.zero, i32 %sign, i32 %subnormal"
        )
        .unwrap();
        writeln!(output, "  %exponent.zero = icmp eq i32 %exponent, 0").unwrap();
        writeln!(
            output,
            "  %finite = select i1 %exponent.zero, i32 %zero.or.subnormal, i32 %normal"
        )
        .unwrap();
        writeln!(output, "  %exponent.special = icmp eq i32 %exponent, 31").unwrap();
        writeln!(
            output,
            "  %result.bits = select i1 %exponent.special, i32 %special, i32 %finite"
        )
        .unwrap();
        writeln!(output, "  %result = bitcast i32 %result.bits to float").unwrap();
        writeln!(output, "  ret float %result").unwrap();
        writeln!(output, "}}\n").unwrap();
    }
    if requirements
        .conversions
        .contains(&FloatConversionKind::F32ToF16RoundTiesEven)
    {
        emit_f32_to_f16_helper(output, attributes);
    }
    if requirements
        .conversions
        .contains(&FloatConversionKind::Bf16ToF32)
    {
        writeln!(
            output,
            "define internal float @__fe2o3_bf16_to_f32_v1(i16 %bits) alwaysinline nounwind{attributes} {{"
        )
        .unwrap();
        writeln!(output, "entry:").unwrap();
        writeln!(output, "  %wide = zext i16 %bits to i32").unwrap();
        writeln!(output, "  %shifted = shl i32 %wide, 16").unwrap();
        writeln!(output, "  %result = bitcast i32 %shifted to float").unwrap();
        writeln!(output, "  ret float %result").unwrap();
        writeln!(output, "}}\n").unwrap();
    }
    if requirements
        .conversions
        .contains(&FloatConversionKind::F32ToBf16RoundTiesEven)
    {
        writeln!(
            output,
            "define internal i16 @__fe2o3_f32_to_bf16_rne_v1(float %value) alwaysinline nounwind{attributes} {{"
        )
        .unwrap();
        writeln!(output, "entry:").unwrap();
        writeln!(output, "  %bits = bitcast float %value to i32").unwrap();
        writeln!(output, "  %exponent = and i32 %bits, 2139095040").unwrap();
        writeln!(output, "  %fraction = and i32 %bits, 8388607").unwrap();
        writeln!(output, "  %special = icmp eq i32 %exponent, 2139095040").unwrap();
        writeln!(output, "  %payload = icmp ne i32 %fraction, 0").unwrap();
        writeln!(output, "  %is.nan = and i1 %special, %payload").unwrap();
        writeln!(output, "  %upper = lshr i32 %bits, 16").unwrap();
        writeln!(output, "  %nan = or i32 %upper, 64").unwrap();
        writeln!(output, "  %lsb = and i32 %upper, 1").unwrap();
        writeln!(output, "  %bias = add i32 32767, %lsb").unwrap();
        writeln!(output, "  %biased = add i32 %bits, %bias").unwrap();
        writeln!(output, "  %rounded = lshr i32 %biased, 16").unwrap();
        writeln!(
            output,
            "  %selected = select i1 %is.nan, i32 %nan, i32 %rounded"
        )
        .unwrap();
        writeln!(output, "  %result = trunc i32 %selected to i16").unwrap();
        writeln!(output, "  ret i16 %result").unwrap();
        writeln!(output, "}}\n").unwrap();
    }
}

fn emit_f32_to_f16_helper(output: &mut dyn fmt::Write, attributes: &str) {
    writeln!(
        output,
        "define internal i16 @__fe2o3_f32_to_f16_rne_v1(float %value) alwaysinline nounwind{attributes} {{"
    )
    .unwrap();
    writeln!(output, "entry:").unwrap();
    writeln!(output, "  %bits = bitcast float %value to i32").unwrap();
    writeln!(output, "  %sign.shift = lshr i32 %bits, 16").unwrap();
    writeln!(output, "  %sign = and i32 %sign.shift, 32768").unwrap();
    writeln!(output, "  %exponent.shift = lshr i32 %bits, 23").unwrap();
    writeln!(output, "  %exponent = and i32 %exponent.shift, 255").unwrap();
    writeln!(output, "  %fraction = and i32 %bits, 8388607").unwrap();
    writeln!(output, "  %payload.shift = lshr i32 %fraction, 13").unwrap();
    writeln!(output, "  %payload.quiet = or i32 %payload.shift, 512").unwrap();
    writeln!(output, "  %nan.payload = or i32 31744, %payload.quiet").unwrap();
    writeln!(output, "  %nan = or i32 %sign, %nan.payload").unwrap();
    writeln!(output, "  %infinity = or i32 %sign, 31744").unwrap();
    writeln!(output, "  %fraction.nonzero = icmp ne i32 %fraction, 0").unwrap();
    writeln!(
        output,
        "  %special.value = select i1 %fraction.nonzero, i32 %nan, i32 %infinity"
    )
    .unwrap();
    writeln!(output, "  %half.exponent = sub i32 %exponent, 112").unwrap();
    writeln!(output, "  %normal.truncated = lshr i32 %fraction, 13").unwrap();
    writeln!(output, "  %normal.remainder = and i32 %fraction, 8191").unwrap();
    writeln!(
        output,
        "  %normal.greater = icmp ugt i32 %normal.remainder, 4096"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.equal = icmp eq i32 %normal.remainder, 4096"
    )
    .unwrap();
    writeln!(output, "  %normal.odd.bits = and i32 %normal.truncated, 1").unwrap();
    writeln!(output, "  %normal.odd = icmp ne i32 %normal.odd.bits, 0").unwrap();
    writeln!(
        output,
        "  %normal.tie.up = and i1 %normal.equal, %normal.odd"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.round.up = or i1 %normal.greater, %normal.tie.up"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.increment = zext i1 %normal.round.up to i32"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.rounded = add i32 %normal.truncated, %normal.increment"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.carry = icmp eq i32 %normal.rounded, 1024"
    )
    .unwrap();
    writeln!(output, "  %normal.carry.i32 = zext i1 %normal.carry to i32").unwrap();
    writeln!(
        output,
        "  %normal.exponent.adjusted = add i32 %half.exponent, %normal.carry.i32"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.overflow = icmp sge i32 %normal.exponent.adjusted, 31"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.exponent.bits = shl i32 %normal.exponent.adjusted, 10"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.fraction = select i1 %normal.carry, i32 0, i32 %normal.rounded"
    )
    .unwrap();
    writeln!(
        output,
        "  %normal.payload = or i32 %normal.exponent.bits, %normal.fraction"
    )
    .unwrap();
    writeln!(output, "  %normal.signed = or i32 %sign, %normal.payload").unwrap();
    writeln!(
        output,
        "  %normal.value = select i1 %normal.overflow, i32 %infinity, i32 %normal.signed"
    )
    .unwrap();
    writeln!(output, "  %raw.shift = sub i32 14, %half.exponent").unwrap();
    writeln!(output, "  %shift.too.large = icmp ugt i32 %raw.shift, 31").unwrap();
    writeln!(
        output,
        "  %safe.shift = select i1 %shift.too.large, i32 31, i32 %raw.shift"
    )
    .unwrap();
    writeln!(output, "  %significand = or i32 %fraction, 8388608").unwrap();
    writeln!(
        output,
        "  %subnormal.truncated = lshr i32 %significand, %safe.shift"
    )
    .unwrap();
    writeln!(output, "  %mask.base = shl i32 1, %safe.shift").unwrap();
    writeln!(output, "  %mask = sub i32 %mask.base, 1").unwrap();
    writeln!(
        output,
        "  %subnormal.remainder = and i32 %significand, %mask"
    )
    .unwrap();
    writeln!(output, "  %half.shift = sub i32 %safe.shift, 1").unwrap();
    writeln!(output, "  %halfway = shl i32 1, %half.shift").unwrap();
    writeln!(
        output,
        "  %subnormal.greater = icmp ugt i32 %subnormal.remainder, %halfway"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.equal = icmp eq i32 %subnormal.remainder, %halfway"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.odd.bits = and i32 %subnormal.truncated, 1"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.odd = icmp ne i32 %subnormal.odd.bits, 0"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.tie.up = and i1 %subnormal.equal, %subnormal.odd"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.round.up = or i1 %subnormal.greater, %subnormal.tie.up"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.increment = zext i1 %subnormal.round.up to i32"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.rounded = add i32 %subnormal.truncated, %subnormal.increment"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.signed = or i32 %sign, %subnormal.rounded"
    )
    .unwrap();
    writeln!(
        output,
        "  %deep.underflow = icmp slt i32 %half.exponent, -10"
    )
    .unwrap();
    writeln!(
        output,
        "  %subnormal.value = select i1 %deep.underflow, i32 %sign, i32 %subnormal.signed"
    )
    .unwrap();
    writeln!(output, "  %is.subnormal = icmp sle i32 %half.exponent, 0").unwrap();
    writeln!(
        output,
        "  %finite = select i1 %is.subnormal, i32 %subnormal.value, i32 %normal.value"
    )
    .unwrap();
    writeln!(output, "  %is.special = icmp eq i32 %exponent, 255").unwrap();
    writeln!(
        output,
        "  %selected = select i1 %is.special, i32 %special.value, i32 %finite"
    )
    .unwrap();
    writeln!(output, "  %result = trunc i32 %selected to i16").unwrap();
    writeln!(output, "  ret i16 %result").unwrap();
    writeln!(output, "}}\n").unwrap();
}

fn constrained_binary_name(op: WidenedFloatBinaryOp) -> &'static str {
    match op {
        WidenedFloatBinaryOp::Add => "llvm.experimental.constrained.fadd.f32",
        WidenedFloatBinaryOp::Subtract => "llvm.experimental.constrained.fsub.f32",
        WidenedFloatBinaryOp::Multiply => "llvm.experimental.constrained.fmul.f32",
        WidenedFloatBinaryOp::Divide => "llvm.experimental.constrained.fdiv.f32",
    }
}

fn narrow_float_helpers(format: NarrowFloatFormat) -> (&'static str, &'static str) {
    match format {
        NarrowFloatFormat::F16 => ("__fe2o3_f16_to_f32_v1", "__fe2o3_f32_to_f16_rne_v1"),
        NarrowFloatFormat::Bf16 => ("__fe2o3_bf16_to_f32_v1", "__fe2o3_f32_to_bf16_rne_v1"),
    }
}

fn constrained_math_name(function: F32MathFunction, target: LoweringTarget) -> &'static str {
    match function {
        F32MathFunction::Sqrt if target == LoweringTarget::Gfx950XnackMinusV1 => "llvm.sqrt.f32",
        F32MathFunction::Sqrt => "llvm.experimental.constrained.sqrt.f32",
        F32MathFunction::FusedMultiplyAdd => "llvm.experimental.constrained.fma.f32",
        F32MathFunction::Floor => "llvm.experimental.constrained.floor.f32",
        F32MathFunction::Ceil => "llvm.experimental.constrained.ceil.f32",
        F32MathFunction::Truncate => "llvm.experimental.constrained.trunc.f32",
        F32MathFunction::RoundTiesEven => "llvm.experimental.constrained.roundeven.f32",
        _ => unreachable!("OCML functions are not LLVM constrained intrinsics"),
    }
}

fn ocml_name(function: F32MathFunction) -> &'static str {
    match function {
        F32MathFunction::Sin => "__ocml_sin_f32",
        F32MathFunction::Cos => "__ocml_cos_f32",
        F32MathFunction::Exp => "__ocml_exp_f32",
        F32MathFunction::Exp2 => "__ocml_exp2_f32",
        F32MathFunction::Ln => "__ocml_log_f32",
        F32MathFunction::Log2 => "__ocml_log2_f32",
        F32MathFunction::Log10 => "__ocml_log10_f32",
        _ => unreachable!("constrained functions are not OCML calls"),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IntrinsicAttribute {
    ReadNone,
    Convergent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IntrinsicDeclaration {
    result: &'static str,
    arguments: &'static str,
    attribute: IntrinsicAttribute,
}

fn emit_compiler_module(
    module: &Module,
    kernels: &[FunctionLowerer<'_>],
    helpers: &[FunctionLowerer<'_>],
    declarations: &[&Function],
    target: LoweringTarget,
) -> Result<String, LoweringErrors> {
    let intrinsics = collect_intrinsic_declarations(kernels.iter().chain(helpers));
    let memcpy_address_spaces = collect_memcpy_declarations(kernels.iter().chain(helpers));
    let float_requirements = FloatRequirements::collect(kernels.iter().chain(helpers));
    let diagnostic_requirements = DiagnosticRequirements::collect(kernels.iter().chain(helpers));
    let has_readnone = intrinsics
        .values()
        .any(|declaration| declaration.attribute == IntrinsicAttribute::ReadNone);
    let has_convergent = intrinsics
        .values()
        .any(|declaration| declaration.attribute == IntrinsicAttribute::Convergent);
    let readnone_attribute = has_readnone.then_some(kernels.len());
    let convergent_attribute = has_convergent.then_some(kernels.len() + usize::from(has_readnone));

    let mut output = CapacityLimitedText::new(MAX_COMPILER_MODULE_TEXT_BYTES);
    writeln!(output, "target triple = \"{AMDGPU_TRIPLE}\"").unwrap();
    if let Some(data_layout) = target.data_layout() {
        writeln!(output, "target datalayout = \"{data_layout}\"").unwrap();
    }
    writeln!(output).unwrap();

    let mut has_lds = false;
    for lowerer in kernels {
        has_lds |= lowerer.emit_workgroup_memory_declarations(&mut output);
    }
    if has_lds {
        writeln!(output).unwrap();
    }

    for (symbol, declaration) in &intrinsics {
        let attribute = match declaration.attribute {
            IntrinsicAttribute::ReadNone => readnone_attribute.expect("readnone attribute"),
            IntrinsicAttribute::Convergent => convergent_attribute.expect("convergent attribute"),
        };
        writeln!(
            output,
            "declare {} @{symbol}({}) #{attribute}",
            declaration.result, declaration.arguments
        )
        .unwrap();
    }
    for (destination, source) in &memcpy_address_spaces {
        let destination = llvm_address_space(*destination);
        let source = llvm_address_space(*source);
        writeln!(
            output,
            "declare void @llvm.memcpy.p{destination}.p{source}.i64(ptr addrspace({destination}) noalias nocapture writeonly, ptr addrspace({source}) noalias nocapture readonly, i64, i1 immarg)"
        )
        .unwrap();
    }
    emit_float_support_declarations(&mut output, &float_requirements, target);
    emit_diagnostic_declarations(&mut output, &diagnostic_requirements);
    for function in declarations {
        writeln!(
            output,
            "declare {} @{}({})",
            llvm_result_type(&function.signature),
            function.id,
            llvm_parameter_types(&function.signature).join(", ")
        )
        .unwrap();
    }
    if !intrinsics.is_empty()
        || !memcpy_address_spaces.is_empty()
        || !declarations.is_empty()
        || !float_requirements.is_empty()
        || !diagnostic_requirements.is_empty()
    {
        writeln!(output).unwrap();
    }

    emit_float_support_definitions(&mut output, &float_requirements, target);

    for (index, lowerer) in kernels.iter().enumerate() {
        lowerer.emit_compiler_module_definition(&mut output, Some(index), Some(index))?;
    }
    for lowerer in helpers {
        lowerer.emit_compiler_module_definition(&mut output, None, None)?;
    }

    for (index, lowerer) in kernels.iter().enumerate() {
        let wave_attribute = lowerer
            .wave_width
            .map_or("", |width| target.wave_target_feature(width));
        let flat_workgroup_size = lowerer
            .flat_workgroup_size()
            .expect("compiler-module kernel requires a workgroup size");
        match lowerer.launch_bounds {
            Some(bounds) => {
                writeln!(
                    output,
                    "attributes #{index} = {{ nounwind \"amdgpu-flat-work-group-size\"=\"{},{}\" \"amdgpu-waves-per-eu\"=\"{},{}\"{wave_attribute}{} }}",
                    bounds.minimum_flat_workgroup_size(),
                    bounds.maximum_flat_workgroup_size(),
                    bounds.minimum_waves_per_execution_unit(),
                    bounds.maximum_waves_per_execution_unit(),
                    target.llvm_function_attributes()
                )
                .unwrap();
            }
            None => {
                writeln!(
                    output,
                    "attributes #{index} = {{ nounwind \"amdgpu-flat-work-group-size\"=\"{flat_workgroup_size},{flat_workgroup_size}\"{wave_attribute}{} }}",
                    target.llvm_function_attributes()
                )
                .unwrap();
            }
        }
    }
    if let Some(index) = readnone_attribute {
        writeln!(
            output,
            "attributes #{index} = {{ nounwind readnone speculatable willreturn }}"
        )
        .unwrap();
    }
    if let Some(index) = convergent_attribute {
        writeln!(output, "attributes #{index} = {{ convergent nounwind }}").unwrap();
    }
    writeln!(output).unwrap();
    for (index, lowerer) in kernels.iter().enumerate() {
        let workgroup_size = lowerer
            .workgroup_size
            .expect("compiler-module kernel requires a workgroup size");
        writeln!(
            output,
            "!{index} = !{{i32 {}, i32 {}, i32 {}}}",
            workgroup_size.x, workgroup_size.y, workgroup_size.z
        )
        .unwrap();
    }
    output.finish(module)
}

struct CapacityLimitedText {
    output: String,
    attempted_bytes: usize,
    max_bytes: usize,
    overflowed: bool,
}

impl CapacityLimitedText {
    fn new(max_bytes: usize) -> Self {
        Self {
            output: String::with_capacity(max_bytes),
            attempted_bytes: 0,
            max_bytes,
            overflowed: false,
        }
    }

    fn finish(self, module: &Module) -> Result<String, LoweringErrors> {
        if self.overflowed {
            return Err(LoweringErrors::one(
                LoweringLocation::module(module),
                LoweringDiagnosticCode::ResourceLimit,
                format!(
                    "compiler-module textual LLVM attempted {} bytes; maximum is {}",
                    self.attempted_bytes, self.max_bytes
                ),
            ));
        }
        Ok(self.output)
    }
}

impl fmt::Write for CapacityLimitedText {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.attempted_bytes = self.attempted_bytes.saturating_add(text.len());
        if self.overflowed || self.attempted_bytes > self.max_bytes {
            self.overflowed = true;
            return Ok(());
        }
        self.output.push_str(text);
        Ok(())
    }
}

fn collect_memcpy_declarations<'a>(
    lowerers: impl Iterator<Item = &'a FunctionLowerer<'a>>,
) -> BTreeSet<(KernelAddressSpace, KernelAddressSpace)> {
    lowerers
        .flat_map(|lowerer| {
            lowerer
                .function
                .body
                .as_ref()
                .expect("definition required")
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
        })
        .filter_map(|operation| {
            let OperationKind::MemoryIntrinsic(MemoryIntrinsicOperation::CopyNonOverlapping {
                element,
                source_address_space,
                destination_address_space,
                ..
            }) = operation.kind
            else {
                return None;
            };
            (element != MemoryElementType::Unit)
                .then_some((destination_address_space, source_address_space))
        })
        .collect()
}

fn collect_intrinsic_declarations<'a>(
    lowerers: impl Iterator<Item = &'a FunctionLowerer<'a>>,
) -> BTreeMap<String, IntrinsicDeclaration> {
    let mut declarations = BTreeMap::new();
    for lowerer in lowerers {
        let body = lowerer.function.body.as_ref().expect("definition required");
        for operation in body.blocks.iter().flat_map(|block| &block.operations) {
            if let OperationKind::Intrinsic(intrinsic) = &operation.kind
                && let IntrinsicKind::InvocationIndex {
                    kind: IndexKind::WorkgroupCount,
                    ..
                } = intrinsic.kind
            {
                insert_intrinsic(
                    &mut declarations,
                    AmdgcnIntrinsic::DispatchPtr,
                    "ptr addrspace(4)",
                    "",
                    IntrinsicAttribute::ReadNone,
                );
            }
            match &operation.kind {
                OperationKind::Intrinsic(_) => {
                    let rank = lowerer.kernel.map_or(1, |kernel| kernel.domain.rank());
                    for dim in [Dim::X, Dim::Y, Dim::Z].into_iter().take(usize::from(rank)) {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::WorkItemId(dim),
                            "i32",
                            "",
                            IntrinsicAttribute::ReadNone,
                        );
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::WorkGroupId(dim),
                            "i32",
                            "",
                            IntrinsicAttribute::ReadNone,
                        );
                    }
                    if rank > 1 {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::DispatchPtr,
                            "ptr addrspace(4)",
                            "",
                            IntrinsicAttribute::ReadNone,
                        );
                    }
                }
                OperationKind::Binary {
                    op: BinaryOp::Checked(operator),
                    lhs,
                    ..
                } => {
                    let scalar = lowerer
                        .value_type(*lhs)
                        .as_scalar()
                        .expect("preflight accepted checked integer operation");
                    let (result, arguments) = checked_binary_intrinsic_signature(scalar);
                    let name = checked_binary_intrinsic_name(*operator, scalar);
                    let previous = declarations.insert(
                        name,
                        IntrinsicDeclaration {
                            result,
                            arguments,
                            attribute: IntrinsicAttribute::ReadNone,
                        },
                    );
                    debug_assert!(previous.is_none_or(|previous| {
                        previous
                            == IntrinsicDeclaration {
                                result,
                                arguments,
                                attribute: IntrinsicAttribute::ReadNone,
                            }
                    }));
                }
                OperationKind::WorkgroupBarrier(_)
                    if !lowerer.target.requires_physical_workgroup_barrier() =>
                {
                    insert_intrinsic(
                        &mut declarations,
                        AmdgcnIntrinsic::SBarrier,
                        "void",
                        "",
                        IntrinsicAttribute::Convergent,
                    )
                }
                OperationKind::WorkgroupBarrier(_) => {}
                OperationKind::Wave(wave) => {
                    if matches!(
                        wave.kind,
                        WaveOperationKind::LaneId
                            | WaveOperationKind::ShuffleIndex { .. }
                            | WaveOperationKind::ReduceF32 { .. }
                            | WaveOperationKind::BroadcastF32 { .. }
                    ) {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntLo,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                        if wave.width == WaveWidth::Wave64 {
                            insert_intrinsic(
                                &mut declarations,
                                AmdgcnIntrinsic::MbcntHi,
                                "i32",
                                "i32, i32",
                                IntrinsicAttribute::ReadNone,
                            );
                        }
                    }
                    if matches!(
                        wave.kind,
                        WaveOperationKind::Ballot { .. }
                            | WaveOperationKind::Any { .. }
                            | WaveOperationKind::All { .. }
                    ) {
                        let (result, intrinsic) = ballot_intrinsic(wave.width);
                        declarations.insert(
                            intrinsic.to_owned(),
                            IntrinsicDeclaration {
                                result,
                                arguments: "i1",
                                attribute: IntrinsicAttribute::Convergent,
                            },
                        );
                    }
                    if matches!(
                        wave.kind,
                        WaveOperationKind::ShuffleIndex { .. }
                            | WaveOperationKind::ReduceF32 { .. }
                            | WaveOperationKind::BroadcastF32 { .. }
                    ) {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::DsBpermute,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::Convergent,
                        );
                    }
                }
                OperationKind::Gfx950LdsTranspose(transpose) => match transpose.kind {
                    Gfx950LdsTransposeOperationKindV1::Current { .. }
                    | Gfx950LdsTransposeOperationKindV1::Publish { .. } => {}
                    Gfx950LdsTransposeOperationKindV1::Stage { .. } => {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntLo,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntHi,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                        for name in ["llvm.uadd.with.overflow.i64", "llvm.umul.with.overflow.i64"] {
                            declarations.insert(
                                name.to_owned(),
                                IntrinsicDeclaration {
                                    result: "{ i64, i1 }",
                                    arguments: "i64, i64",
                                    attribute: IntrinsicAttribute::ReadNone,
                                },
                            );
                        }
                    }
                    Gfx950LdsTransposeOperationKindV1::Read { format, .. } => {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntLo,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntHi,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                        let intrinsic = match format {
                            Gfx950LdsTransposeFormatV1::Fp4E2M1 => AmdgcnIntrinsic::DsReadTr4B64,
                            Gfx950LdsTransposeFormatV1::Fp8E4M3 => AmdgcnIntrinsic::DsReadTr8B64,
                        };
                        insert_intrinsic(
                            &mut declarations,
                            intrinsic,
                            "<2 x i32>",
                            "ptr addrspace(3) nocapture",
                            IntrinsicAttribute::Convergent,
                        );
                    }
                },
                OperationKind::Matrix(matrix) => match &matrix.kind {
                    MatrixOperationKind::MultiplyAccumulate { .. } => insert_intrinsic(
                        &mut declarations,
                        AmdgcnIntrinsic::MfmaF32M16N16K16Bf16,
                        "<4 x float>",
                        "<4 x i16>, <4 x i16>, <4 x float>, i32, i32, i32",
                        IntrinsicAttribute::Convergent,
                    ),
                    MatrixOperationKind::ScaledMultiplyAccumulate { .. } => insert_intrinsic(
                        &mut declarations,
                        AmdgcnIntrinsic::MfmaScaleF32M16N16K128F8F6F4V8I32,
                        "<4 x float>",
                        "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32",
                        IntrinsicAttribute::Convergent,
                    ),
                    MatrixOperationKind::LdsLoad { .. } | MatrixOperationKind::LdsStore { .. } => {
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntLo,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                        insert_intrinsic(
                            &mut declarations,
                            AmdgcnIntrinsic::MbcntHi,
                            "i32",
                            "i32, i32",
                            IntrinsicAttribute::ReadNone,
                        );
                    }
                },
                _ => {}
            }
        }
    }
    declarations
}

fn insert_intrinsic(
    declarations: &mut BTreeMap<String, IntrinsicDeclaration>,
    intrinsic: AmdgcnIntrinsic,
    result: &'static str,
    arguments: &'static str,
    attribute: IntrinsicAttribute,
) {
    let previous = declarations.insert(
        intrinsic.llvm_name().to_owned(),
        IntrinsicDeclaration {
            result,
            arguments,
            attribute,
        },
    );
    debug_assert!(previous.is_none_or(|previous| {
        previous
            == IntrinsicDeclaration {
                result,
                arguments,
                attribute,
            }
    }));
}

fn llvm_parameter_types(signature: &Signature) -> Vec<&'static str> {
    signature.parameters.iter().map(llvm_type).collect()
}

fn llvm_result_type(signature: &Signature) -> &'static str {
    match signature.results.as_slice() {
        [] => "void",
        [result] => llvm_type(result),
        _ => unreachable!("compiler-module preflight rejected multi-value returns"),
    }
}

fn is_safe_symbol(symbol: &str) -> bool {
    let mut bytes = symbol.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn is_reserved_float_support_symbol(symbol: &str) -> bool {
    symbol.starts_with("__fe2o3_ir_float_v1_")
        || matches!(
            symbol,
            "__fe2o3_f16_to_f32_v1"
                | "__fe2o3_f32_to_f16_rne_v1"
                | "__fe2o3_bf16_to_f32_v1"
                | "__fe2o3_f32_to_bf16_rne_v1"
                | "__ocml_sin_f32"
                | "__ocml_cos_f32"
                | "__ocml_exp_f32"
                | "__ocml_exp2_f32"
                | "__ocml_log_f32"
                | "__ocml_log2_f32"
                | "__ocml_log10_f32"
        )
}

fn validate_capabilities(
    location: LoweringLocation,
    capabilities: &BTreeSet<TargetCapability>,
    owner: &str,
    target: LoweringTarget,
) -> Result<Option<WaveWidth>, LoweringErrors> {
    let mut wave_width = None;
    for capability in capabilities {
        match capability {
            TargetCapability::WorkgroupMemory
            | TargetCapability::WorkgroupBarrier
            | TargetCapability::DynamicWorkgroupMemory
            | TargetCapability::Subgroups => {}
            TargetCapability::Float16 | TargetCapability::BFloat16
                if target.supports_narrow_float() => {}
            TargetCapability::SubgroupSize(32 | 64) => {}
            TargetCapability::WaveWidth(width) => wave_width = Some(*width),
            TargetCapability::Atomic {
                width_bits,
                address_space,
                max_scope,
            } if supported_atomic_capability(*width_bits, *address_space, *max_scope) => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_gfx942_inline_assembly()
                    && namespace == AMDGPU_GFX942_INLINE_ASSEMBLY_CAPABILITY_NAMESPACE
                    && name == AMDGPU_GFX942_INLINE_ASSEMBLY_CAPABILITY_NAME => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_bf16_matrix_mfma()
                    && namespace == MATRIX_CAPABILITY_NAMESPACE
                    && name == BF16_F32_M16N16K16_CAPABILITY => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_gfx942_matrix_lds()
                    && namespace == MATRIX_CAPABILITY_NAMESPACE
                    && name == LDS_TILE_16X16_XOR4_CAPABILITY => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_gfx950_scaled_matrix()
                    && namespace == MATRIX_CAPABILITY_NAMESPACE
                    && matches!(
                        name.as_str(),
                        SCALED_FP4_E2M1_F32_M16N16K128_CAPABILITY
                            | SCALED_FP4_E2M1_FP8_E4M3_F32_M16N16K128_CAPABILITY
                            | SCALED_FP8_E4M3_F32_M16N16K128_CAPABILITY
                    ) => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_gfx942_diagnostics()
                    && namespace == AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAMESPACE
                    && name == AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAME => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_amdgpu_diagnostics()
                    && namespace == AMDGPU_DIAGNOSTICS_CAPABILITY_NAMESPACE
                    && name == AMDGPU_DIAGNOSTICS_CAPABILITY_NAME => {}
            TargetCapability::Extension { namespace, name }
                if target.supports_gfx942_xnack_minus_binding()
                    && namespace == AMDGPU_EXACT_TARGET_CAPABILITY_NAMESPACE
                    && name == AMDGPU_GFX942_XNACK_MINUS_TARGET_CAPABILITY_NAME => {}
            TargetCapability::Extension { namespace, name }
                if target.exact_target_binding() == Some(name.as_str())
                    && namespace == AMDGPU_EXACT_TARGET_CAPABILITY_NAMESPACE => {}
            TargetCapability::Extension { namespace, name }
                if target == LoweringTarget::Gfx942XnackMinusV1
                    && matches!(
                        namespace.as_str(),
                        MATRIX_SOURCE_ABI_OBSERVATION_NAMESPACE_V2
                            | MATRIX_PROJECTED_KERNARG_POLICY_NAMESPACE_V1
                    )
                    && is_lower_hex_digest(name) => {}
            _ => {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedCapability,
                    format!("G1 does not lower {owner} capability {capability:?}"),
                ));
            }
        }
    }
    Ok(wave_width)
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_exact_kernel_binding(
    module: &Module,
    kernel: &Kernel,
    entry: &Function,
    exact_target: &'static str,
) -> Result<(), LoweringErrors> {
    let required = TargetCapability::Extension {
        namespace: AMDGPU_EXACT_TARGET_CAPABILITY_NAMESPACE.to_owned(),
        name: exact_target.to_owned(),
    };
    for (location, owner, capabilities) in [
        (
            LoweringLocation::module(module),
            "module",
            &module.required_capabilities,
        ),
        (
            LoweringLocation::kernel(module, kernel),
            "kernel",
            &kernel.required_capabilities,
        ),
        (
            LoweringLocation::function(module, kernel, entry),
            "entry function",
            &entry.required_capabilities,
        ),
    ] {
        if !capabilities.contains(&required) {
            return Err(LoweringErrors::one(
                location,
                LoweringDiagnosticCode::UnsupportedCapability,
                format!("exact {exact_target} lowering requires {required:?} on the {owner}"),
            ));
        }
    }
    Ok(())
}

fn validate_matrix_frontend_abi_binding(
    module: &Module,
    kernel: &Kernel,
    entry: &Function,
    target: LoweringTarget,
) -> Result<(), LoweringErrors> {
    let bindings = entry
        .body
        .iter()
        .flat_map(|body| &body.blocks)
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OperationKind::Matrix(matrix) => matrix.frontend_binding.as_ref(),
            _ => None,
        })
        .collect::<Vec<_>>();
    let owners = [
        &module.required_capabilities,
        &kernel.required_capabilities,
        &entry.required_capabilities,
    ];
    let has_claim_capability = owners.iter().any(|capabilities| {
        capabilities.iter().any(|capability| {
            matches!(
                capability,
                TargetCapability::Extension { namespace, .. }
                    if matches!(
                        namespace.as_str(),
                        MATRIX_SOURCE_ABI_OBSERVATION_NAMESPACE_V2
                            | MATRIX_PROJECTED_KERNARG_POLICY_NAMESPACE_V1
                    )
            )
        })
    });
    if bindings.is_empty() && !has_claim_capability {
        return Ok(());
    }
    if target != LoweringTarget::Gfx942XnackMinusV1 {
        return Err(LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::UnsupportedCapability,
            "matrix source ABI observations and projected kernarg policy are admitted only by exact gfx942:xnack- lowering",
        ));
    }
    require_exact_kernel_binding(
        module,
        kernel,
        entry,
        AMDGPU_GFX942_XNACK_MINUS_TARGET_CAPABILITY_NAME,
    )?;
    let [binding] = bindings.as_slice() else {
        return Err(LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::UnsupportedCapability,
            "exactly one matrix operation must carry the structured rustc source ABI observation and projected kernarg policy",
        ));
    };
    binding.validate().map_err(|reason| {
        LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::UnsupportedCapability,
            reason,
        )
    })?;
    let required = binding.capabilities();
    for capabilities in owners {
        if required
            .iter()
            .any(|capability| !capabilities.contains(capability))
        {
            return Err(LoweringErrors::one(
                LoweringLocation::function(module, kernel, entry),
                LoweringDiagnosticCode::UnsupportedCapability,
                "matrix source-observation and projected-policy digests must be bound on the module, kernel, and entry function",
            ));
        }
    }
    let expected = [vec![Type::Scalar(ScalarType::Bf16); 8], vec![Type::F32; 4]].concat();
    if entry.signature.parameters != expected || !entry.signature.results.is_empty() {
        return Err(LoweringErrors::one(
            LoweringLocation::function(module, kernel, entry),
            LoweringDiagnosticCode::UnsupportedParameter,
            "the matrix projected kernarg policy requires exactly 8 BF16 and 4 F32 parameters and no result",
        ));
    }
    Ok(())
}

fn validate_launch(
    module: &Module,
    kernel: &Kernel,
    target: LoweringTarget,
) -> Result<WorkgroupSize, LoweringErrors> {
    match kernel.domain {
        LaunchDomain::D1 {
            x: LaunchExtent::Static(_) | LaunchExtent::Dynamic,
        } => {}
        LaunchDomain::D2 {
            x: LaunchExtent::Static(_),
            y: LaunchExtent::Static(_),
        }
        | LaunchDomain::D3 {
            x: LaunchExtent::Static(_),
            y: LaunchExtent::Static(_),
            z: LaunchExtent::Static(_),
        } => {}
        LaunchDomain::D2 { .. } | LaunchDomain::D3 { .. }
            if matches!(
                target,
                LoweringTarget::Baseline
                    | LoweringTarget::Gfx942StrictFloatV1
                    | LoweringTarget::Gfx942XnackMinusV1
                    | LoweringTarget::Gfx950XnackMinusV1
            ) => {}
        LaunchDomain::D2 { .. } | LaunchDomain::D3 { .. } => {
            return Err(LoweringErrors::one(
                LoweringLocation::kernel(module, kernel),
                LoweringDiagnosticCode::UnsupportedLaunchDomain,
                "the authenticated specialized profile requires its exact static launch domain",
            ));
        }
    }

    let Some(size) = kernel.workgroup_size else {
        return Err(LoweringErrors::one(
            LoweringLocation::kernel(module, kernel),
            LoweringDiagnosticCode::MissingWorkgroupSize,
            "G1 requires a statically declared workgroup size",
        ));
    };
    let Some(flat_size) = checked_flat_workgroup_size(size) else {
        return Err(LoweringErrors::one(
            LoweringLocation::kernel(module, kernel),
            LoweringDiagnosticCode::UnsupportedWorkgroupSize,
            "workgroup dimensions overflow the flat workgroup size",
        ));
    };
    if flat_size == 0 || flat_size > MAX_G1_FLAT_WORKGROUP_SIZE {
        return Err(LoweringErrors::one(
            LoweringLocation::kernel(module, kernel),
            LoweringDiagnosticCode::UnsupportedWorkgroupSize,
            format!(
                "flat workgroup size {flat_size} exceeds the G1 limit of {MAX_G1_FLAT_WORKGROUP_SIZE}"
            ),
        ));
    }
    Ok(size)
}

fn checked_flat_workgroup_size(size: WorkgroupSize) -> Option<u32> {
    size.x.checked_mul(size.y)?.checked_mul(size.z)
}

#[derive(Clone)]
enum ValueBinding {
    Value {
        llvm_name: String,
        ty: Type,
    },
    Slice {
        data_name: String,
        length_name: String,
        ty: Type,
    },
}

impl ValueBinding {
    fn value(&self) -> Option<(&str, &Type)> {
        match self {
            Self::Value { llvm_name, ty } => Some((llvm_name, ty)),
            Self::Slice { .. } => None,
        }
    }
}

fn control_flow_emission_plan(function: &Function) -> (IndexedControlFlow, Vec<bool>) {
    let control_flow = analyze_control_flow(function)
        .expect("verify_module established bounded structural control flow");
    let body = function.body.as_ref().expect("definition required");
    let mut split_edges = vec![false; control_flow.edge_count()];
    let mut target_counts = vec![0usize; control_flow.block_count()];
    let mut touched_targets = Vec::new();
    for block in &body.blocks {
        let outgoing = control_flow
            .outgoing_edges(block.id)
            .expect("indexed source block");
        touched_targets.clear();
        for edge_index in outgoing.clone() {
            let target = control_flow.edge_target(edge_index).expect("indexed edge");
            let target_position = control_flow
                .block_position(target)
                .expect("indexed target block");
            if target_counts[target_position] == 0 {
                touched_targets.push(target_position);
            }
            target_counts[target_position] += 1;
        }
        for edge_index in outgoing.clone() {
            let target = control_flow.edge_target(edge_index).expect("indexed edge");
            let target_position = control_flow
                .block_position(target)
                .expect("indexed target block");
            if body.blocks[target_position].parameters.is_empty() {
                continue;
            }
            let duplicate_target = target_counts[target_position] > 1;
            let critical = outgoing.len() > 1
                && control_flow
                    .incoming_edges(target)
                    .expect("indexed target block")
                    .len()
                    > 1;
            split_edges[edge_index] = duplicate_target || critical;
        }
        for target in &touched_targets {
            target_counts[*target] = 0;
        }
    }
    (control_flow, split_edges)
}

struct FunctionLowerer<'a> {
    module: &'a Module,
    kernel: Option<&'a Kernel>,
    function: &'a Function,
    symbol: &'a str,
    workgroup_size: Option<WorkgroupSize>,
    wave_width: Option<WaveWidth>,
    target: LoweringTarget,
    launch_bounds: Option<Gfx942LaunchBoundsV1>,
    call_symbols: Option<&'a BTreeMap<FunctionId, String>>,
    bindings: BTreeMap<ValueId, ValueBinding>,
    control_flow: IndexedControlFlow,
    split_edges: Vec<bool>,
}

fn matrix_frontend_binding(function: &Function) -> Option<&MatrixFrontendBindingV2> {
    function
        .body
        .iter()
        .flat_map(|body| &body.blocks)
        .flat_map(|block| &block.operations)
        .find_map(|operation| match &operation.kind {
            OperationKind::Matrix(matrix) => matrix.frontend_binding.as_ref(),
            _ => None,
        })
}

fn emit_matrix_projected_kernarg_policy(
    output: &mut dyn fmt::Write,
    policy: &MatrixProjectedKernargPolicyV1,
) {
    writeln!(
        output,
        "; fe2o3.projected-kernarg-policy.v1 sha256={}",
        lower_hex(&policy.digest)
    )
    .unwrap();
    writeln!(
        output,
        "; fe2o3.projected-kernarg explicit-size={} implicit-bytes={} segment-size={} segment-align={} source=compiler-policy-not-rustc-observation",
        policy.explicit_argument_size,
        policy.implicit_argument_bytes,
        policy.kernarg_segment_size,
        policy.kernarg_segment_alignment,
    )
    .unwrap();
    for (index, parameter) in policy.parameters.iter().enumerate() {
        let kind = match parameter.element {
            MatrixElement::Bf16 => "bf16",
            MatrixElement::F32 => "f32",
            MatrixElement::Fp4E2M1 => "fp4-e2m1",
            MatrixElement::Fp8E4M3 => "fp8-e4m3",
        };
        writeln!(
            output,
            "; fe2o3.projected-kernarg.param index={index} source={} lane={} type={kind} offset={} size={} align={}",
            parameter.source,
            parameter.lane,
            parameter.offset,
            parameter.size,
            parameter.alignment,
        )
        .unwrap();
    }
}

fn lower_hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(encoded, "{byte:02x}").unwrap();
    }
    encoded
}

impl<'a> FunctionLowerer<'a> {
    fn flat_workgroup_size(&self) -> Option<u32> {
        self.workgroup_size.and_then(checked_flat_workgroup_size)
    }

    fn emit_logical_global_id(&self, output: &mut dyn fmt::Write, result: &str) {
        let kernel = self
            .kernel
            .expect("global invocation index requires a kernel");
        let workgroup = self
            .workgroup_size
            .expect("global invocation index requires a kernel workgroup size");
        if kernel.domain.rank() == 1 {
            writeln!(
                output,
                "  {result}.local.i32 = call i32 @{}()",
                AmdgcnIntrinsic::WorkItemId(Dim::X).llvm_name()
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.group.i32 = call i32 @{}()",
                AmdgcnIntrinsic::WorkGroupId(Dim::X).llvm_name()
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.local = zext i32 {result}.local.i32 to i64"
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.group = zext i32 {result}.group.i32 to i64"
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.base = mul i64 {result}.group, {}",
                workgroup.x
            )
            .unwrap();
            writeln!(output, "  {result} = add i64 {result}.base, {result}.local").unwrap();
            return;
        }

        for (dim, extent) in [
            (Dim::X, workgroup.x),
            (Dim::Y, workgroup.y),
            (Dim::Z, workgroup.z),
        ]
        .into_iter()
        .take(usize::from(kernel.domain.rank()))
        {
            let suffix = dim.suffix();
            writeln!(
                output,
                "  {result}.{suffix}.local.i32 = call i32 @{}()",
                AmdgcnIntrinsic::WorkItemId(dim).llvm_name()
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.{suffix}.group.i32 = call i32 @{}()",
                AmdgcnIntrinsic::WorkGroupId(dim).llvm_name()
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.{suffix}.local = zext i32 {result}.{suffix}.local.i32 to i64"
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.{suffix}.group = zext i32 {result}.{suffix}.group.i32 to i64"
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.{suffix}.base = mul i64 {result}.{suffix}.group, {extent}"
            )
            .unwrap();
            writeln!(
                output,
                "  {result}.{suffix} = add i64 {result}.{suffix}.base, {result}.{suffix}.local"
            )
            .unwrap();
        }
        writeln!(
            output,
            "  {result}.dispatch = call ptr addrspace(4) @{}()",
            AmdgcnIntrinsic::DispatchPtr.llvm_name()
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.grid.x.ptr = getelementptr inbounds i8, ptr addrspace(4) {result}.dispatch, i64 12"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.grid.x.i32 = load i32, ptr addrspace(4) {result}.grid.x.ptr, align 4"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.grid.x = zext i32 {result}.grid.x.i32 to i64"
        )
        .unwrap();
        if kernel.domain.rank() == 2 {
            writeln!(
                output,
                "  {result}.row = mul i64 {result}.y, {result}.grid.x"
            )
            .unwrap();
            writeln!(output, "  {result} = add i64 {result}.row, {result}.x").unwrap();
            return;
        }
        writeln!(
            output,
            "  {result}.grid.y.ptr = getelementptr inbounds i8, ptr addrspace(4) {result}.dispatch, i64 16"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.grid.y.i32 = load i32, ptr addrspace(4) {result}.grid.y.ptr, align 4"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.grid.y = zext i32 {result}.grid.y.i32 to i64"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.plane = mul i64 {result}.z, {result}.grid.y"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.plane_row = add i64 {result}.plane, {result}.y"
        )
        .unwrap();
        writeln!(
            output,
            "  {result}.plane_row_scaled = mul i64 {result}.plane_row, {result}.grid.x"
        )
        .unwrap();
        writeln!(
            output,
            "  {result} = add i64 {result}.plane_row_scaled, {result}.x"
        )
        .unwrap();
    }

    fn new(
        module: &'a Module,
        kernel: &'a Kernel,
        function: &'a Function,
        workgroup_size: WorkgroupSize,
        wave_width: Option<WaveWidth>,
        target: LoweringTarget,
    ) -> Self {
        let (control_flow, split_edges) = control_flow_emission_plan(function);
        Self {
            module,
            kernel: Some(kernel),
            function,
            symbol: kernel.id.as_str(),
            workgroup_size: Some(workgroup_size),
            wave_width,
            target,
            launch_bounds: None,
            call_symbols: None,
            bindings: BTreeMap::new(),
            control_flow,
            split_edges,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compiler_module_kernel(
        module: &'a Module,
        kernel: &'a Kernel,
        function: &'a Function,
        workgroup_size: WorkgroupSize,
        wave_width: Option<WaveWidth>,
        call_symbols: &'a BTreeMap<FunctionId, String>,
        target: LoweringTarget,
        launch_bounds: Option<Gfx942LaunchBoundsV1>,
    ) -> Self {
        let (control_flow, split_edges) = control_flow_emission_plan(function);
        Self {
            module,
            kernel: Some(kernel),
            function,
            symbol: kernel.id.as_str(),
            workgroup_size: Some(workgroup_size),
            wave_width,
            target,
            launch_bounds,
            call_symbols: Some(call_symbols),
            bindings: BTreeMap::new(),
            control_flow,
            split_edges,
        }
    }

    fn compiler_module_device_function(
        module: &'a Module,
        function: &'a Function,
        wave_width: Option<WaveWidth>,
        call_symbols: &'a BTreeMap<FunctionId, String>,
        target: LoweringTarget,
    ) -> Self {
        let (control_flow, split_edges) = control_flow_emission_plan(function);
        Self {
            module,
            kernel: None,
            function,
            symbol: function.id.as_str(),
            workgroup_size: None,
            wave_width,
            target,
            launch_bounds: None,
            call_symbols: Some(call_symbols),
            bindings: BTreeMap::new(),
            control_flow,
            split_edges,
        }
    }

    fn block(&self, block: BlockId) -> &BasicBlock {
        let position = self
            .control_flow
            .block_position(block)
            .expect("verified block has an index");
        &self
            .function
            .body
            .as_ref()
            .expect("definition required")
            .blocks[position]
    }

    fn function_location(&self) -> LoweringLocation {
        self.kernel.map_or_else(
            || LoweringLocation::device_function(self.module, self.function),
            |kernel| LoweringLocation::function(self.module, kernel, self.function),
        )
    }

    fn block_location(&self, block: BlockId) -> LoweringLocation {
        self.kernel.map_or_else(
            || LoweringLocation::device_block(self.module, self.function, block),
            |kernel| LoweringLocation::block(self.module, kernel, self.function, block),
        )
    }

    fn operation_location(&self, block: BlockId, operation: usize) -> LoweringLocation {
        self.kernel.map_or_else(
            || LoweringLocation::device_operation(self.module, self.function, block, operation),
            |kernel| {
                LoweringLocation::operation(self.module, kernel, self.function, block, operation)
            },
        )
    }

    fn declares_capability(&self, required: &TargetCapability) -> bool {
        self.module
            .required_capabilities
            .iter()
            .chain(&self.function.required_capabilities)
            .chain(
                self.kernel
                    .into_iter()
                    .flat_map(|kernel| &kernel.required_capabilities),
            )
            .any(|declared| declared == required)
    }

    fn declares_operation_capability(&self, required: &TargetCapability) -> bool {
        if self.declares_capability(required) {
            return true;
        }
        match required {
            TargetCapability::Extension { namespace, name }
                if namespace == AMDGPU_DIAGNOSTICS_CAPABILITY_NAMESPACE
                    && name == AMDGPU_DIAGNOSTICS_CAPABILITY_NAME =>
            {
                self.declares_capability(&TargetCapability::Extension {
                    namespace: AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAMESPACE.to_owned(),
                    name: AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAME.to_owned(),
                })
            }
            TargetCapability::Extension { namespace, name }
                if namespace == AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAMESPACE
                    && name == AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAME =>
            {
                self.declares_capability(&TargetCapability::Extension {
                    namespace: AMDGPU_DIAGNOSTICS_CAPABILITY_NAMESPACE.to_owned(),
                    name: AMDGPU_DIAGNOSTICS_CAPABILITY_NAME.to_owned(),
                })
            }
            _ => false,
        }
    }

    fn validate_operation_capability_declarations(
        &self,
        operation: &Operation,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let is_float = matches!(
            &operation.kind,
            OperationKind::Call { callee, arguments }
                if FloatOperation::from_intrinsic_call(callee, arguments).is_some()
        );
        let is_diagnostic = matches!(
            &operation.kind,
            OperationKind::Call { callee, arguments }
                if AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments).is_some()
        );
        let is_inline_assembly = matches!(operation.kind, OperationKind::InlineAssembly(_));
        let is_matrix = matches!(operation.kind, OperationKind::Matrix(_));
        let is_gfx950_collective_or_lds_transpose = matches!(
            operation.kind,
            OperationKind::Gfx950LdsTranspose(_)
                | OperationKind::Wave(WaveOperation {
                    kind: WaveOperationKind::ReduceF32 { .. }
                        | WaveOperationKind::BroadcastF32 { .. },
                    ..
                })
        );
        if !is_float
            && !is_diagnostic
            && !is_inline_assembly
            && !is_matrix
            && !is_gfx950_collective_or_lds_transpose
            && !matches!(
                &operation.kind,
                OperationKind::Fence(_)
                    | OperationKind::WorkgroupBarrier(_)
                    | OperationKind::WorkgroupMemory(_)
            )
        {
            return Ok(());
        }
        for required in operation
            .required_capabilities()
            .into_iter()
            .filter(|capability| {
                matches!(
                    capability,
                    TargetCapability::WorkgroupMemory
                        | TargetCapability::DynamicWorkgroupMemory
                        | TargetCapability::WorkgroupBarrier
                        | TargetCapability::Subgroups
                        | TargetCapability::SubgroupSize(_)
                        | TargetCapability::WaveWidth(_)
                        | TargetCapability::Float16
                        | TargetCapability::BFloat16
                ) || matches!(
                    capability,
                    TargetCapability::Extension { namespace, name }
                        if namespace == AMDGPU_GFX942_INLINE_ASSEMBLY_CAPABILITY_NAMESPACE
                            && name == AMDGPU_GFX942_INLINE_ASSEMBLY_CAPABILITY_NAME
                ) || matches!(
                    capability,
                    TargetCapability::Extension { namespace, .. }
                        if namespace == MATRIX_CAPABILITY_NAMESPACE
                ) || matches!(
                    capability,
                    TargetCapability::Extension { namespace, name }
                        if namespace == AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAMESPACE
                            && name == AMDGPU_GFX942_DIAGNOSTICS_CAPABILITY_NAME
                ) || matches!(
                    capability,
                    TargetCapability::Extension { namespace, name }
                        if namespace == AMDGPU_DIAGNOSTICS_CAPABILITY_NAMESPACE
                            && name == AMDGPU_DIAGNOSTICS_CAPABILITY_NAME
                )
            })
        {
            if !self.declares_operation_capability(&required) {
                return Err(LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::UnsupportedCapability,
                    format!(
                        "AMDGPU lowering requires an explicit {required:?} capability declaration"
                    ),
                ));
            }
        }
        Ok(())
    }

    fn validate_lds_addressability(&self) -> Result<(), LoweringErrors> {
        let body = self.function.body.as_ref().expect("definition required");
        let mut static_end = 0u64;
        for block in &body.blocks {
            for (operation_index, operation) in block.operations.iter().enumerate() {
                let (alignment, static_bytes) = match &operation.kind {
                    OperationKind::WorkgroupMemory(memory) => {
                        let bytes = match memory.extent {
                            WorkgroupMemoryExtent::Static(elements) => {
                                let element_bytes = amdgpu_lds_element_bytes(&memory.element)
                                    .expect("operation preflight accepted the LDS element type");
                                Some(u64::from(elements) * element_bytes)
                            }
                            WorkgroupMemoryExtent::Dynamic
                            | WorkgroupMemoryExtent::DynamicAtLeast(_) => None,
                        };
                        (u64::from(memory.alignment), bytes)
                    }
                    OperationKind::Gfx950LdsTranspose(transpose)
                        if let Gfx950LdsTransposeOperationKindV1::Current { format } =
                            transpose.kind =>
                    {
                        (64, Some(u64::from(format.lds_bytes())))
                    }
                    _ => continue,
                };
                let padding = (alignment - static_end % alignment) % alignment;
                static_end = static_end
                    .checked_add(padding)
                    .expect("u32 LDS alignments cannot overflow u64");
                if let Some(bytes) = static_bytes {
                    static_end = static_end
                        .checked_add(bytes)
                        .expect("u32 LDS extents cannot overflow u64");
                }
                if static_end > u64::from(u32::MAX) {
                    return Err(LoweringErrors::one(
                        self.operation_location(block.id, operation_index),
                        LoweringDiagnosticCode::UnsupportedWorkgroupMemory,
                        format!(
                            "LDS declarations require at least {static_end} bytes, exceeding the AMDGPU 32-bit LDS address space"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_parameters(&mut self) -> Result<(), LoweringErrors> {
        let body = self.function.body.as_ref().expect("definition required");
        for (index, (value, ty)) in body
            .parameters
            .iter()
            .copied()
            .zip(&self.function.signature.parameters)
            .enumerate()
        {
            let location = self.function_location();
            self.validate_narrow_type_capability(ty, &location)?;
            match ty {
                Type::Scalar(scalar) => {
                    if !supported_scalar(*scalar, self.target) {
                        return Err(LoweringErrors::one(
                            location,
                            LoweringDiagnosticCode::UnsupportedType,
                            format!("unsupported kernel parameter {index}: {ty:?}"),
                        ));
                    }
                    self.bindings.insert(
                        value,
                        ValueBinding::Value {
                            llvm_name: format!("%arg{index}"),
                            ty: ty.clone(),
                        },
                    );
                }
                Type::Pointer(_) if self.kernel.is_none() => {
                    validate_device_pointer(ty, &location, self.target)?;
                    self.bindings.insert(
                        value,
                        ValueBinding::Value {
                            llvm_name: format!("%arg{index}"),
                            ty: ty.clone(),
                        },
                    );
                }
                Type::Pointer(pointer)
                    if self.kernel.is_some()
                        && pointer.address_space == KernelAddressSpace::Global
                        && supported_memory_type(&pointer.pointee, self.target) =>
                {
                    self.bindings.insert(
                        value,
                        ValueBinding::Value {
                            llvm_name: format!("%arg{index}"),
                            ty: ty.clone(),
                        },
                    );
                }
                Type::Pointer(pointer) => {
                    let code = if pointer.address_space != KernelAddressSpace::Global {
                        LoweringDiagnosticCode::UnsupportedAddressSpace
                    } else {
                        LoweringDiagnosticCode::UnsupportedType
                    };
                    return Err(LoweringErrors::one(
                        location,
                        code,
                        format!("unsupported kernel parameter {index}: {ty:?}"),
                    ));
                }
                Type::Slice(slice)
                    if self.kernel.is_some()
                        && slice.address_space == KernelAddressSpace::Global
                        && supported_memory_type(&slice.element, self.target) =>
                {
                    self.bindings.insert(
                        value,
                        ValueBinding::Slice {
                            data_name: format!("%arg{index}.data"),
                            length_name: format!("%arg{index}.len"),
                            ty: ty.clone(),
                        },
                    );
                }
                Type::Slice(slice) => {
                    let code = if slice.address_space != KernelAddressSpace::Global {
                        LoweringDiagnosticCode::UnsupportedAddressSpace
                    } else {
                        LoweringDiagnosticCode::UnsupportedType
                    };
                    return Err(LoweringErrors::one(
                        location,
                        code,
                        format!("unsupported kernel parameter {index}: {ty:?}"),
                    ));
                }
                _ => {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedParameter,
                        format!("unsupported kernel parameter {index}: {ty:?}"),
                    ));
                }
            }
        }
        for block in &body.blocks {
            for parameter in &block.parameters {
                let location = self.block_location(block.id);
                self.validate_narrow_type_capability(&parameter.ty, &location)?;
                match &parameter.ty {
                    Type::Scalar(scalar) if supported_scalar(*scalar, self.target) => {
                        self.bindings.insert(
                            parameter.id,
                            ValueBinding::Value {
                                llvm_name: value_name(parameter.id),
                                ty: parameter.ty.clone(),
                            },
                        );
                    }
                    Type::Pointer(_) => {
                        validate_device_pointer(&parameter.ty, &location, self.target)?;
                        self.bindings.insert(
                            parameter.id,
                            ValueBinding::Value {
                                llvm_name: value_name(parameter.id),
                                ty: parameter.ty.clone(),
                            },
                        );
                    }
                    Type::Slice(slice)
                        if self.kernel.is_some()
                            && slice.address_space == KernelAddressSpace::Global
                            && supported_memory_type(&slice.element, self.target) =>
                    {
                        self.bindings.insert(
                            parameter.id,
                            ValueBinding::Slice {
                                data_name: format!("{}.data", value_name(parameter.id)),
                                length_name: format!("{}.len", value_name(parameter.id)),
                                ty: parameter.ty.clone(),
                            },
                        );
                    }
                    Type::Slice(slice) => {
                        let code = if slice.address_space != KernelAddressSpace::Global {
                            LoweringDiagnosticCode::UnsupportedAddressSpace
                        } else {
                            LoweringDiagnosticCode::UnsupportedType
                        };
                        return Err(LoweringErrors::one(
                            location,
                            code,
                            format!(
                                "unsupported block parameter {}: {:?}",
                                parameter.id, parameter.ty
                            ),
                        ));
                    }
                    _ => {
                        return Err(LoweringErrors::one(
                            location,
                            LoweringDiagnosticCode::UnsupportedType,
                            format!(
                                "unsupported block parameter {}: {:?}",
                                parameter.id, parameter.ty
                            ),
                        ));
                    }
                }
            }
        }
        for block in &body.blocks {
            for operation in &block.operations {
                for result in &operation.results {
                    self.validate_narrow_type_capability(
                        &result.ty,
                        &self.block_location(block.id),
                    )?;
                    let llvm_name = match &operation.kind {
                        OperationKind::Constant(constant) => {
                            constant_value(constant).unwrap_or_else(|| value_name(result.id))
                        }
                        _ => value_name(result.id),
                    };
                    self.bindings.insert(
                        result.id,
                        ValueBinding::Value {
                            llvm_name,
                            ty: result.ty.clone(),
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_narrow_type_capability(
        &self,
        ty: &Type,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let scalar = match ty {
            Type::Scalar(scalar) => Some(*scalar),
            Type::Pointer(pointer) => pointer.pointee.as_scalar(),
            Type::Slice(slice) => slice.element.as_scalar(),
            Type::Unit => None,
        };
        let required = match scalar {
            Some(ScalarType::F16) => Some(TargetCapability::Float16),
            Some(ScalarType::Bf16) => Some(TargetCapability::BFloat16),
            _ => None,
        };
        if let Some(required) = required
            && !self.declares_capability(&required)
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedCapability,
                format!("AMDGPU lowering requires an explicit {required:?} type capability"),
            ));
        }
        Ok(())
    }

    fn validate_block(&mut self, block: &BasicBlock) -> Result<(), LoweringErrors> {
        for (index, operation) in block.operations.iter().enumerate() {
            self.validate_operation(block.id, index, operation)?;
        }
        self.validate_terminator(
            block.id,
            block
                .terminator
                .as_ref()
                .expect("verify_module required it"),
        )
    }

    fn validate_block_arguments(&self) -> Result<(), LoweringErrors> {
        let body = self.function.body.as_ref().expect("definition required");
        let entry = body
            .blocks
            .first()
            .expect("verify_module required an entry block");
        for block in body
            .blocks
            .iter()
            .filter(|block| !block.parameters.is_empty())
        {
            let location = self.block_location(block.id);
            if block.id == entry.id {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedBlockArguments,
                    "G1 cannot materialize entry-block parameters because the initial entry edge has no SSA arguments",
                ));
            }

            if self
                .control_flow
                .incoming_edges(block.id)
                .expect("indexed block")
                .is_empty()
            {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedBlockArguments,
                    "G1 cannot materialize block parameters without an incoming CFG edge",
                ));
            }
        }
        Ok(())
    }

    fn validate_operation(
        &mut self,
        block: BlockId,
        index: usize,
        operation: &Operation,
    ) -> Result<(), LoweringErrors> {
        let location = self.operation_location(block, index);
        self.validate_operation_capability_declarations(operation, &location)?;
        match &operation.kind {
            OperationKind::Constant(constant) => {
                validate_constant(constant, self.target).map_err(|message| {
                    LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedConstant,
                        message,
                    )
                })?;
            }
            OperationKind::Intrinsic(intrinsic)
                if self.kernel.is_some()
                    && (matches!(
                        intrinsic.kind,
                        IntrinsicKind::InvocationIndex {
                            kind: IndexKind::Global,
                            axis: Axis::X,
                        } | IntrinsicKind::InvocationIndex {
                            kind: IndexKind::Local
                                | IndexKind::Workgroup
                                | IndexKind::WorkgroupSize
                                | IndexKind::WorkgroupCount,
                            axis: Axis::X,
                        }
                    )) => {}
            OperationKind::MemoryIntrinsic(intrinsic) => {
                validate_memory_intrinsic(intrinsic, &location, self.target)?;
            }
            OperationKind::Binary { op, lhs, .. } => {
                let ty = self.value_type(*lhs);
                if matches!(op, BinaryOp::Checked(_)) {
                    if !ty.as_scalar().is_some_and(ScalarType::is_integer) {
                        return Err(LoweringErrors::one(
                            location,
                            LoweringDiagnosticCode::UnsupportedOperation,
                            format!("gfx942 does not lower {op:?} for {ty:?}"),
                        ));
                    }
                    return Ok(());
                }
                if !supported_binary(*op, ty, self.target) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedOperation,
                        format!("G1 does not lower {op:?} for {ty:?}"),
                    ));
                }
            }
            OperationKind::Unary { op, operand } => {
                let ty = self.value_type(*operand);
                if !supported_unary(*op, ty) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedOperation,
                        format!("G1 does not lower {op:?} for {ty:?}"),
                    ));
                }
            }
            OperationKind::Compare { lhs, .. } => {
                let ty = self.value_type(*lhs);
                if !ty.as_scalar().is_some_and(|scalar| {
                    scalar == ScalarType::Bool
                        || supported_integer(scalar)
                        || (scalar == ScalarType::F32
                            && !matches!(self.target, LoweringTarget::Baseline))
                }) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedOperation,
                        "G1 lowers comparisons only for supported scalar values",
                    ));
                }
            }
            OperationKind::Select { true_value, .. } => {
                let ty = self.value_type(*true_value);
                if !ty.as_scalar().is_some_and(|scalar| {
                    scalar == ScalarType::Bool
                        || supported_integer(scalar)
                        || scalar == ScalarType::F32
                }) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedOperation,
                        "G1 lowers select only for supported scalar values",
                    ));
                }
            }
            OperationKind::Cast { kind, value, to } => {
                let from = self.value_type(*value);
                validate_cast(*kind, from, to, self.target).map_err(|message| {
                    LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedCast,
                        message,
                    )
                })?;
            }
            OperationKind::SliceLength { slice } => {
                let binding = self.bindings.get(slice).expect("verified operand");
                let ValueBinding::Slice { .. } = binding else {
                    unreachable!("verify_module checked slice_length")
                };
            }
            OperationKind::SliceData { slice } => {
                let binding = self.bindings.get(slice).expect("verified operand");
                let ValueBinding::Slice { .. } = binding else {
                    unreachable!("verify_module checked slice_data")
                };
            }
            OperationKind::GetElementPointer { base, .. } => {
                validate_pointer(self.value_type(*base), &location, self.target)?;
            }
            OperationKind::Load { pointer, access } => {
                validate_memory_access(
                    self.value_type(*pointer),
                    access.address_space,
                    &location,
                    self.target,
                )?;
            }
            OperationKind::GuardedLoad {
                pointer, access, ..
            } => {
                validate_memory_access(
                    self.value_type(*pointer),
                    access.address_space,
                    &location,
                    self.target,
                )?;
            }
            OperationKind::Store {
                pointer, access, ..
            } => {
                validate_memory_access(
                    self.value_type(*pointer),
                    access.address_space,
                    &location,
                    self.target,
                )?;
            }
            OperationKind::Fence(_) => {}
            OperationKind::WorkgroupBarrier(_) if self.kernel.is_some() => {}
            OperationKind::WorkgroupBarrier(_) => {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedBarrier,
                    "compiler-module helpers cannot contain kernel-context workgroup barriers",
                ));
            }
            OperationKind::WorkgroupMemory(memory) => {
                if self.kernel.is_none() {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedWorkgroupMemory,
                        "compiler-module helpers cannot own kernel-context LDS declarations",
                    ));
                }
                if !supported_memory_type(&memory.element, self.target) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedWorkgroupMemory,
                        format!(
                            "AMDGPU LDS lowering does not support element type {:?}",
                            memory.element
                        ),
                    ));
                }
                let natural_alignment = amdgpu_lds_element_bytes(&memory.element)
                    .expect("supported LDS types have a fixed AMDGPU size");
                if u64::from(memory.alignment) < natural_alignment {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedWorkgroupMemory,
                        format!(
                            "LDS element type {:?} requires alignment {natural_alignment}, found {}",
                            memory.element, memory.alignment
                        ),
                    ));
                }
            }
            OperationKind::InlineAssembly(assembly) => {
                self.validate_inline_assembly(operation, assembly, &location)?;
            }
            OperationKind::Matrix(matrix) => {
                self.validate_matrix(matrix, &location)?;
            }
            OperationKind::Gfx950LdsTranspose(transpose) => {
                self.validate_gfx950_lds_transpose(transpose, &location)?;
            }
            OperationKind::Wave(wave) => self.validate_wave(wave, &location)?,
            OperationKind::Atomic(atomic) => self.validate_atomic(atomic, &location)?,
            OperationKind::Barrier(_) => {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedBarrier,
                    "legacy barriers lack the convergence evidence required by AMDGPU lowering",
                ));
            }
            OperationKind::Alloca {
                address_space: KernelAddressSpace::Workgroup,
                ..
            } => {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedWorkgroupMemory,
                    "workgroup Alloca is ambiguous; use explicit WorkgroupMemory",
                ));
            }
            OperationKind::Alloca {
                element,
                count: None,
                address_space: KernelAddressSpace::Private,
                alignment,
            } => {
                if !supported_private_memory_type(element, self.target) {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedType,
                        format!("unsupported private allocation element type {element:?}"),
                    ));
                }
                let natural_alignment = amdgpu_private_element_alignment(element)
                    .expect("supported private allocation types have a fixed alignment");
                if u64::from(*alignment) < natural_alignment {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedOperation,
                        format!(
                            "private allocation element {element:?} requires alignment {natural_alignment}, found {alignment}"
                        ),
                    ));
                }
            }
            OperationKind::Call { callee, arguments } => {
                if let Some(diagnostic) =
                    AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
                {
                    self.validate_diagnostic(&diagnostic, &location)?;
                } else if let Some(float) = FloatOperation::from_intrinsic_call(callee, arguments) {
                    self.validate_float(&float, &location)?;
                } else if self.call_symbols.is_some() {
                    self.validate_call(callee, arguments, operation, &location)?;
                } else {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedOperation,
                        format!("G1 does not lower {:?}", operation.kind),
                    ));
                }
            }
            OperationKind::Intrinsic(_) | OperationKind::Alloca { .. } => {
                return Err(LoweringErrors::one(
                    location,
                    LoweringDiagnosticCode::UnsupportedOperation,
                    format!("G1 does not lower {:?}", operation.kind),
                ));
            }
        }
        Ok(())
    }

    fn validate_matrix(
        &self,
        matrix: &MatrixOperation,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        if self.kernel.is_none() {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedMatrixOperation,
                "compiler-module helpers cannot contain kernel-context matrix operations",
            ));
        }
        if self.wave_width != Some(WaveWidth::Wave64) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedMatrixOperation,
                "matrix V1 requires an exact wave64 capability",
            ));
        }
        let flat_workgroup_size = self
            .flat_workgroup_size()
            .expect("kernel matrix operations have a validated workgroup size");
        if flat_workgroup_size == 0
            || !flat_workgroup_size.is_multiple_of(WaveWidth::Wave64.lanes())
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedMatrixOperation,
                format!(
                    "matrix V1 requires a nonzero flat workgroup size that is a multiple of 64, found {flat_workgroup_size}"
                ),
            ));
        }
        let supported = match &matrix.kind {
            MatrixOperationKind::MultiplyAccumulate { profile, .. } => {
                self.target.supports_bf16_matrix_mfma()
                    && *profile == MatrixMultiplyProfile::bf16_f32_m16n16k16_wave64()
            }
            MatrixOperationKind::ScaledMultiplyAccumulate { profile, .. } => {
                self.target.supports_gfx950_scaled_matrix()
                    && matches!(
                        *profile,
                        value if value
                            == MatrixMultiplyProfile::fp4_e2m1_f32_m16n16k128_wave64()
                            || value
                                == MatrixMultiplyProfile::fp8_e4m3_f32_m16n16k128_wave64()
                    )
            }
            MatrixOperationKind::LdsLoad { profile, .. }
            | MatrixOperationKind::LdsStore { profile, .. } => {
                self.target.supports_gfx942_matrix_lds() && profile.is_supported_v1()
            }
        };
        if !supported || matrix.active_lanes != 64 {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedMatrixOperation,
                "matrix operation is outside the exact V1 shape and active-lane profile",
            ));
        }
        if let MatrixOperationKind::LdsLoad { base, profile }
        | MatrixOperationKind::LdsStore { base, profile, .. } = &matrix.kind
        {
            let allocation = self
                .function
                .body
                .iter()
                .flat_map(|body| &body.blocks)
                .flat_map(|block| &block.operations)
                .find_map(|operation| {
                    (operation.results.first().map(|result| result.id) == Some(*base))
                        .then_some(match &operation.kind {
                            OperationKind::WorkgroupMemory(memory) => Some(memory),
                            _ => None,
                        })
                        .flatten()
                });
            let Some(allocation) = allocation else {
                return Err(LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::UnsupportedMatrixOperation,
                    "matrix LDS base is not the direct result of an authenticated workgroup-memory allocation",
                ));
            };
            let required_elements = profile.required_elements();
            match allocation.extent.guaranteed_elements() {
                Some(elements) if elements >= required_elements => {}
                Some(elements) => {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedMatrixOperation,
                        format!(
                            "matrix LDS allocation guarantees {elements} elements but requires at least {required_elements}"
                        ),
                    ));
                }
                None => {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedMatrixOperation,
                        "matrix LDS lowering requires an authenticated allocation extent",
                    ));
                }
            }
            if allocation.alignment < profile.required_alignment() {
                return Err(LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::UnsupportedMatrixOperation,
                    "matrix LDS allocation does not meet the profile alignment",
                ));
            }
        }
        Ok(())
    }

    fn validate_diagnostic(
        &self,
        diagnostic: &AmdGpuDiagnosticOperation,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        if !self.target.supports_amdgpu_diagnostics() {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedDiagnosticOperation,
                "device diagnostics require an exact supported AMDGPU target profile",
            ));
        }
        if self.target == LoweringTarget::Gfx950XnackMinusV1
            && !matches!(
                diagnostic,
                AmdGpuDiagnosticOperation::Trap | AmdGpuDiagnosticOperation::AssertFail { .. }
            )
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedDiagnosticOperation,
                "the exact gfx950 profile admits only terminating trap diagnostics",
            ));
        }
        let require_constant = |value, field: &str| {
            self.constant_u32(value).ok_or_else(|| {
                LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::UnsupportedDiagnosticOperation,
                    format!("{field} must be a compile-time u32 constant"),
                )
            })
        };
        match diagnostic {
            AmdGpuDiagnosticOperation::ProfilingMarker { marker } => {
                let marker = require_constant(*marker, "profiling marker")?;
                if marker > u32::from(u16::MAX) {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedDiagnosticOperation,
                        "profiling marker exceeds the gfx942 V1 16-bit range",
                    ));
                }
            }
            AmdGpuDiagnosticOperation::Print { format_id, .. } => {
                if require_constant(*format_id, "diagnostic format identity")? == 0 {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedDiagnosticOperation,
                        "diagnostic format identity must be nonzero",
                    ));
                }
            }
            AmdGpuDiagnosticOperation::AssertFail { site_id, line } => {
                if require_constant(*site_id, "assertion site identity")? == 0
                    || require_constant(*line, "assertion source line")? == 0
                {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedDiagnosticOperation,
                        "assertion site identity and source line must be nonzero",
                    ));
                }
            }
            AmdGpuDiagnosticOperation::Clock32
            | AmdGpuDiagnosticOperation::Trap
            | AmdGpuDiagnosticOperation::DebugTrap => {}
        }
        Ok(())
    }

    fn constant_u32(&self, value: ValueId) -> Option<u32> {
        let operation = self
            .function
            .body
            .as_ref()?
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|operation| operation.results.iter().any(|result| result.id == value))?;
        match operation.kind {
            OperationKind::Constant(Constant::U32(value)) => Some(value),
            _ => None,
        }
    }

    fn bounded_u32_source_lane(&self, value: ValueId, width: u32) -> bool {
        if width == 0 || !width.is_power_of_two() || width > 64 {
            return false;
        }
        if self.constant_u32(value).is_some_and(|lane| lane < width) {
            return true;
        }
        let Some(OperationKind::Binary {
            op: BinaryOp::BitAnd,
            lhs,
            rhs,
        }) = self
            .function
            .body
            .iter()
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .find(|operation| operation.results.iter().any(|result| result.id == value))
            .map(|operation| &operation.kind)
        else {
            return false;
        };
        self.constant_u32(*lhs).is_some_and(|mask| mask < width)
            || self.constant_u32(*rhs).is_some_and(|mask| mask < width)
    }

    fn validate_inline_assembly(
        &self,
        operation: &Operation,
        assembly: &InlineAssembly,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        if !self.target.supports_gfx942_inline_assembly()
            || assembly.target != InlineAssemblyTarget::AmdGpuGfx942
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedInlineAssembly,
                "inline assembly is admitted only by the authenticated gfx942 lowering profile",
            ));
        }
        let Some(instruction) = gfx942_assembly_instruction(&assembly.mnemonic) else {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedAssemblyInstruction,
                format!(
                    "gfx942 inline assembly does not admit instruction {:?}",
                    assembly.mnemonic
                ),
            ));
        };
        if !assembly.declared_effects.is_empty()
            || !assembly.options.contains(&AssemblyOption::NoMemory)
            || assembly.options.contains(&AssemblyOption::ReadOnly)
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::AssemblyEffectMismatch,
                "the bounded gfx942 assembly subset is exactly NoMemory and effect-free",
            ));
        }
        let expected_operand_count = instruction.input_count + 1;
        if operation.results.len() != 1 || assembly.operands.len() != expected_operand_count {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::AssemblyOperandMismatch,
                format!(
                    "{} requires one output and {} inputs",
                    instruction.mnemonic, instruction.input_count
                ),
            ));
        }
        let output = &assembly.operands[0];
        if output.constraint != instruction.constraint
            || output.kind != (AssemblyOperandKind::Output { result_index: 0 })
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::AssemblyOperandMismatch,
                format!(
                    "{} output must be result zero with {:?} constraint",
                    instruction.mnemonic, instruction.constraint
                ),
            ));
        }
        let result_type = &operation.results[0].ty;
        if !is_i32_register_type(result_type) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::AssemblyOperandMismatch,
                format!(
                    "{} output requires i32 or u32, found {result_type:?}",
                    instruction.mnemonic
                ),
            ));
        }
        for (index, operand) in assembly.operands[1..].iter().enumerate() {
            let AssemblyOperandKind::Input(value) = operand.kind else {
                return Err(LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::AssemblyOperandMismatch,
                    format!(
                        "{} input {} must be a distinct SSA input role",
                        instruction.mnemonic, index
                    ),
                ));
            };
            let ty = self.value_type(value);
            if operand.constraint != instruction.constraint
                || !is_i32_register_type(ty)
                || ty != result_type
            {
                return Err(LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::AssemblyOperandMismatch,
                    format!(
                        "{} input {} requires {:?} with exact type {result_type:?}, found {:?} with {ty:?}",
                        instruction.mnemonic, index, instruction.constraint, operand.constraint
                    ),
                ));
            }
        }
        Ok(())
    }

    fn validate_call(
        &self,
        callee: &FunctionId,
        _arguments: &[ValueId],
        _operation: &Operation,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let call_symbols = self
            .call_symbols
            .expect("compiler-module call validation requires a symbol table");
        if call_symbols.contains_key(callee) {
            return Ok(());
        }

        Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedOperation,
            format!("compiler-module calls cannot target kernel entry function {callee}"),
        ))
    }

    fn validate_terminator(
        &self,
        block: BlockId,
        terminator: &Terminator,
    ) -> Result<(), LoweringErrors> {
        let location = self.block_location(block);
        match terminator {
            Terminator::Branch { .. }
            | Terminator::ConditionalBranch { .. }
            | Terminator::IntegerSwitch { .. } => Ok(()),
            Terminator::Switch {
                selector, cases, ..
            } => {
                let selector_type = self.value_type(*selector);
                let Some(scalar) = selector_type
                    .as_scalar()
                    .filter(|scalar| supported_integer(*scalar))
                else {
                    return Err(LoweringErrors::one(
                        location,
                        LoweringDiagnosticCode::UnsupportedTerminator,
                        "legacy switch selector must have a fixed-width integer type",
                    ));
                };
                let width = llvm_width(scalar);
                if width < 64 {
                    let limit = 1_u64 << width;
                    if let Some(case) = cases.iter().find(|case| case.value >= limit) {
                        return Err(LoweringErrors::one(
                            location,
                            LoweringDiagnosticCode::UnsupportedTerminator,
                            format!(
                                "legacy switch case {} is not representable in its i{width} selector",
                                case.value
                            ),
                        ));
                    }
                }
                Ok(())
            }
            Terminator::Return { values } if values.is_empty() => Ok(()),
            Terminator::Return { .. } if self.call_symbols.is_some() => Ok(()),
            Terminator::Unreachable => Ok(()),
            Terminator::Return { .. } => Err(LoweringErrors::one(
                location,
                LoweringDiagnosticCode::UnsupportedTerminator,
                format!("G1 does not lower {terminator:?}"),
            )),
        }
    }

    fn incoming_edges(&self, target: BlockId) -> Vec<(BlockId, usize, &[ValueId])> {
        self.control_flow
            .incoming_edges(target)
            .expect("indexed target block")
            .iter()
            .copied()
            .map(|edge_index| {
                let edge = self
                    .control_flow
                    .edge(edge_index)
                    .expect("indexed incoming edge");
                (
                    self.control_flow
                        .edge_source(edge_index)
                        .expect("indexed incoming source"),
                    edge.ordinal(),
                    self.control_flow.edge_arguments(self.function, edge_index),
                )
            })
            .collect()
    }

    fn edge_index(&self, predecessor: BlockId, ordinal: usize, target: BlockId) -> usize {
        let outgoing = self
            .control_flow
            .outgoing_edges(predecessor)
            .expect("indexed predecessor");
        let edge_index = outgoing.start + ordinal;
        assert!(edge_index < outgoing.end, "indexed edge ordinal");
        assert_eq!(
            self.control_flow.edge_target(edge_index),
            Some(target),
            "indexed edge target"
        );
        edge_index
    }

    fn edge_target_label(&self, predecessor: BlockId, ordinal: usize, target: BlockId) -> String {
        if self.split_edges[self.edge_index(predecessor, ordinal, target)] {
            edge_label(predecessor, ordinal, target)
        } else {
            block_label(target)
        }
    }

    fn phi_predecessor_label(
        &self,
        predecessor: BlockId,
        ordinal: usize,
        target: BlockId,
    ) -> String {
        if self.split_edges[self.edge_index(predecessor, ordinal, target)] {
            edge_label(predecessor, ordinal, target)
        } else {
            self.block_exit_label(predecessor)
        }
    }

    fn block_exit_label(&self, block: BlockId) -> String {
        self.function
            .body
            .as_ref()
            .and_then(|body| body.blocks.iter().find(|candidate| candidate.id == block))
            .and_then(|block| {
                block
                    .operations
                    .iter()
                    .enumerate()
                    .filter_map(|(operation_index, operation)| match &operation.kind {
                        OperationKind::GuardedLoad { .. } => {
                            Some(guarded_load_merge_label(block.id, operation_index))
                        }
                        OperationKind::Gfx950LdsTranspose(transpose)
                            if matches!(
                                transpose.kind,
                                Gfx950LdsTransposeOperationKindV1::Stage { .. }
                            ) =>
                        {
                            let result = operation.results.first().expect("verified stage result");
                            let prefix = format!("v{}.transpose", result.id.0);
                            Some(gfx950_stage_load_merge_label(&prefix, 31))
                        }
                        _ => None,
                    })
                    .next_back()
            })
            .unwrap_or_else(|| block_label(block))
    }

    fn value_type(&self, value: ValueId) -> &Type {
        match self
            .bindings
            .get(&value)
            .expect("verify_module checked value")
        {
            ValueBinding::Value { ty, .. } | ValueBinding::Slice { ty, .. } => ty,
        }
    }

    fn value(&self, value: ValueId) -> (&str, &Type) {
        self.bindings
            .get(&value)
            .and_then(ValueBinding::value)
            .expect("validated scalar or pointer value")
    }

    fn validate_gfx950_lds_transpose(
        &self,
        transpose: &Gfx950LdsTransposeOperationV1,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        if !self.target.supports_gfx950_collectives_and_lds_transpose() {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedOperation,
                "gfx950 LDS transpose requires the exact gfx950:xnack- lowering entry point",
            ));
        }
        if self.kernel.is_none()
            || self.wave_width != Some(WaveWidth::Wave64)
            || self.workgroup_size != Some(WorkgroupSize::new(64, 1, 1))
            || transpose.width != WaveWidth::Wave64
            || transpose.active_lanes != 64
            || transpose.convergence.scope() != SynchronizationScope::Workgroup
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedOperation,
                "gfx950 LDS transpose requires one fully active Wave64 kernel entry with exact workgroup size [64, 1, 1] and workgroup-uniform convergence",
            ));
        }
        if let Gfx950LdsTransposeOperationKindV1::Stage { source_slice, .. } = transpose.kind {
            let valid_source = matches!(
                self.value_type(source_slice),
                Type::Slice(slice)
                    if *slice.element == Type::Scalar(ScalarType::U8)
                        && slice.address_space == KernelAddressSpace::Global
                        && slice.access == fe2o3_kernel_ir::AccessMode::ReadOnly
            );
            if !valid_source {
                return Err(LoweringErrors::one(
                    location.clone(),
                    LoweringDiagnosticCode::UnsupportedOperation,
                    "gfx950 LDS transpose staging requires an exact global read-only u8 slice",
                ));
            }
        }
        Ok(())
    }

    fn validate_wave(
        &self,
        wave: &WaveOperation,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        match wave.kind {
            WaveOperationKind::ReduceF32 { tile_width, .. } => {
                if !self.target.supports_gfx950_collectives_and_lds_transpose()
                    || self.kernel.is_none()
                    || wave.width != WaveWidth::Wave64
                    || wave.active_lanes != 64
                    || tile_width == 0
                    || !tile_width.is_power_of_two()
                    || tile_width > 64
                {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedWaveOperation,
                        "gfx950 f32 reduction requires the exact gfx950:xnack- kernel profile, one fully active Wave64, and a power-of-two tile width no larger than 64",
                    ));
                }
            }
            WaveOperationKind::BroadcastF32 { tile_width, .. } => {
                if !self.target.supports_gfx950_collectives_and_lds_transpose()
                    || self.kernel.is_none()
                    || wave.width != WaveWidth::Wave64
                    || wave.active_lanes != 64
                    || tile_width == 0
                    || !tile_width.is_power_of_two()
                    || tile_width > 64
                {
                    return Err(LoweringErrors::one(
                        location.clone(),
                        LoweringDiagnosticCode::UnsupportedWaveOperation,
                        "gfx950 f32 broadcast requires the exact gfx950:xnack- kernel profile, one fully active Wave64, and a power-of-two tile width no larger than 64",
                    ));
                }
            }
            _ => {}
        }
        if let WaveOperationKind::BroadcastF32 {
            source_lane,
            tile_width,
            ..
        } = wave.kind
            && !self.bounded_u32_source_lane(source_lane, tile_width)
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedWaveOperation,
                "gfx950 f32 broadcast requires a statically bounded tile-local source lane",
            ));
        }
        let Some(flat_workgroup_size) = self.flat_workgroup_size() else {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedWaveOperation,
                "compiler-module helpers cannot contain kernel-context wave operations",
            ));
        };
        if self.wave_width != Some(wave.width) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedWaveOperation,
                format!(
                    "wave operation requires an exact {:?} capability on the module, kernel, or entry function",
                    wave.width
                ),
            ));
        }
        if flat_workgroup_size == 0 || !flat_workgroup_size.is_multiple_of(wave.width.lanes()) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedWaveOperation,
                format!(
                    "full-wave execution requires flat workgroup size {} to be a multiple of {}",
                    flat_workgroup_size,
                    wave.width.lanes()
                ),
            ));
        }
        Ok(())
    }

    fn validate_float(
        &self,
        float: &FloatOperation,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        if matches!(self.target, LoweringTarget::Baseline) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedFloatOperation,
                "floating-point contracts require the strict gfx942 lowering entry point",
            ));
        }
        if let FloatOperation::F32Math {
            function,
            implementation,
            ..
        } = float
            && function.required_implementation() != *implementation
        {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedFloatOperation,
                format!("gfx942 refuses {function:?} with implementation {implementation:?}"),
            ));
        }
        Ok(())
    }

    fn emit(&self) -> Result<String, LoweringErrors> {
        let mut output = String::new();
        writeln!(output, "target triple = \"{AMDGPU_TRIPLE}\"").unwrap();
        if let Some(data_layout) = self.target.data_layout() {
            writeln!(output, "target datalayout = \"{data_layout}\"").unwrap();
        }
        if let Some(binding) = matrix_frontend_binding(self.function) {
            emit_matrix_projected_kernarg_policy(&mut output, &binding.projected_kernarg);
        }
        writeln!(output).unwrap();
        let float_requirements = FloatRequirements::collect(std::iter::once(self));
        let diagnostic_requirements = DiagnosticRequirements::collect(std::iter::once(self));
        let has_workgroup_barrier = self.has_workgroup_barrier();
        let has_matrix_lds = self.has_matrix_kind(|kind| {
            matches!(
                kind,
                MatrixOperationKind::LdsLoad { .. } | MatrixOperationKind::LdsStore { .. }
            )
        });
        let has_matrix_mfma = self
            .has_matrix_kind(|kind| matches!(kind, MatrixOperationKind::MultiplyAccumulate { .. }));
        let has_scaled_matrix_mfma = self.has_matrix_kind(|kind| {
            matches!(kind, MatrixOperationKind::ScaledMultiplyAccumulate { .. })
        });
        let has_wave_lane_id = self.has_wave_kind(|kind| {
            matches!(
                kind,
                WaveOperationKind::LaneId
                    | WaveOperationKind::ShuffleIndex { .. }
                    | WaveOperationKind::ReduceF32 { .. }
                    | WaveOperationKind::BroadcastF32 { .. }
            )
        });
        let has_transpose_lane_id = self.has_gfx950_transpose_kind(|kind| {
            matches!(
                kind,
                Gfx950LdsTransposeOperationKindV1::Stage { .. }
                    | Gfx950LdsTransposeOperationKindV1::Read { .. }
            )
        });
        let has_lane_id = has_matrix_lds || has_wave_lane_id || has_transpose_lane_id;
        let has_ballot = self.has_wave_kind(|kind| {
            matches!(
                kind,
                WaveOperationKind::Ballot { .. }
                    | WaveOperationKind::Any { .. }
                    | WaveOperationKind::All { .. }
            )
        });
        let has_shuffle = self.has_wave_kind(|kind| {
            matches!(
                kind,
                WaveOperationKind::ShuffleIndex { .. }
                    | WaveOperationKind::ReduceF32 { .. }
                    | WaveOperationKind::BroadcastF32 { .. }
            )
        });
        let has_gfx950_stage = self.has_gfx950_transpose_kind(|kind| {
            matches!(kind, Gfx950LdsTransposeOperationKindV1::Stage { .. })
        });
        let has_gfx950_read_b4 = self.has_gfx950_transpose_kind(|kind| {
            matches!(
                kind,
                Gfx950LdsTransposeOperationKindV1::Read {
                    format: Gfx950LdsTransposeFormatV1::Fp4E2M1,
                    ..
                }
            )
        });
        let has_gfx950_read_b8 = self.has_gfx950_transpose_kind(|kind| {
            matches!(
                kind,
                Gfx950LdsTransposeOperationKindV1::Read {
                    format: Gfx950LdsTransposeFormatV1::Fp8E4M3,
                    ..
                }
            )
        });
        let has_convergent_operation = has_workgroup_barrier
            || has_matrix_mfma
            || has_scaled_matrix_mfma
            || self.has_wave_kind(|_| true)
            || self.has_gfx950_transpose_kind(|_| true);
        if self.emit_workgroup_memory_declarations(&mut output) {
            writeln!(output).unwrap();
        }
        let invocation_intrinsics = collect_intrinsic_declarations(std::iter::once(self));
        writeln!(
            output,
            "declare i32 @{}() #1",
            AmdgcnIntrinsic::WorkItemId(Dim::X).llvm_name()
        )
        .unwrap();
        writeln!(
            output,
            "declare i32 @{}() #1",
            AmdgcnIntrinsic::WorkGroupId(Dim::X).llvm_name()
        )
        .unwrap();
        for (symbol, declaration) in invocation_intrinsics {
            if symbol == AmdgcnIntrinsic::DispatchPtr.llvm_name() {
                debug_assert_eq!(declaration.result, "ptr addrspace(4)");
                debug_assert_eq!(declaration.arguments, "");
                debug_assert_eq!(declaration.attribute, IntrinsicAttribute::ReadNone);
                writeln!(output, "declare ptr addrspace(4) @{symbol}() #1").unwrap();
            } else if (symbol.starts_with("llvm.amdgcn.workitem.id.")
                || symbol.starts_with("llvm.amdgcn.workgroup.id."))
                && symbol != AmdgcnIntrinsic::WorkItemId(Dim::X).llvm_name()
                && symbol != AmdgcnIntrinsic::WorkGroupId(Dim::X).llvm_name()
            {
                debug_assert_eq!(declaration.result, "i32");
                debug_assert_eq!(declaration.arguments, "");
                debug_assert_eq!(declaration.attribute, IntrinsicAttribute::ReadNone);
                writeln!(output, "declare i32 @{symbol}() #1").unwrap();
            }
        }
        if has_workgroup_barrier && !self.target.requires_physical_workgroup_barrier() {
            writeln!(
                output,
                "declare void @{}() #2",
                AmdgcnIntrinsic::SBarrier.llvm_name()
            )
            .unwrap();
        }
        if has_lane_id {
            writeln!(
                output,
                "declare i32 @{}(i32, i32) #1",
                AmdgcnIntrinsic::MbcntLo.llvm_name()
            )
            .unwrap();
            if self.wave_width == Some(WaveWidth::Wave64) {
                writeln!(
                    output,
                    "declare i32 @{}(i32, i32) #1",
                    AmdgcnIntrinsic::MbcntHi.llvm_name()
                )
                .unwrap();
            }
        }
        if has_ballot {
            let (ty, intrinsic) = match self.wave_width {
                Some(WaveWidth::Wave32) => ("i32", AmdgcnIntrinsic::Ballot32),
                Some(WaveWidth::Wave64) => ("i64", AmdgcnIntrinsic::Ballot64),
                None => unreachable!("wave preflight required an exact width"),
            };
            writeln!(output, "declare {ty} @{}(i1) #2", intrinsic.llvm_name()).unwrap();
        }
        if has_shuffle {
            writeln!(
                output,
                "declare i32 @{}(i32, i32) #2",
                AmdgcnIntrinsic::DsBpermute.llvm_name()
            )
            .unwrap();
        }
        if has_gfx950_stage {
            writeln!(
                output,
                "declare {{ i64, i1 }} @llvm.uadd.with.overflow.i64(i64, i64) #1"
            )
            .unwrap();
            writeln!(
                output,
                "declare {{ i64, i1 }} @llvm.umul.with.overflow.i64(i64, i64) #1"
            )
            .unwrap();
        }
        if has_gfx950_read_b4 {
            writeln!(
                output,
                "declare <2 x i32> @{}(ptr addrspace(3) nocapture) #2",
                AmdgcnIntrinsic::DsReadTr4B64.llvm_name()
            )
            .unwrap();
        }
        if has_gfx950_read_b8 {
            writeln!(
                output,
                "declare <2 x i32> @{}(ptr addrspace(3) nocapture) #2",
                AmdgcnIntrinsic::DsReadTr8B64.llvm_name()
            )
            .unwrap();
        }
        if has_matrix_mfma {
            writeln!(
                output,
                "declare <4 x float> @{}(<4 x i16>, <4 x i16>, <4 x float>, i32, i32, i32) #2",
                AmdgcnIntrinsic::MfmaF32M16N16K16Bf16.llvm_name()
            )
            .unwrap();
        }
        if has_scaled_matrix_mfma {
            writeln!(
                output,
                "declare <4 x float> @{}(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #2",
                AmdgcnIntrinsic::MfmaScaleF32M16N16K128F8F6F4V8I32.llvm_name()
            )
            .unwrap();
        }
        emit_diagnostic_declarations(&mut output, &diagnostic_requirements);
        emit_float_support_declarations(&mut output, &float_requirements, self.target);
        writeln!(output).unwrap();

        emit_float_support_definitions(&mut output, &float_requirements, self.target);

        write!(output, "define amdgpu_kernel void @{}(", self.symbol).unwrap();
        let parameters = self.llvm_parameters()?;
        write!(output, "{}", parameters.join(", ")).unwrap();
        writeln!(output, ") #0 !reqd_work_group_size !0 {{").unwrap();

        self.emit_body(&mut output)?;
        writeln!(output, "}}\n").unwrap();
        let wave_attribute = self
            .wave_width
            .map_or("", |width| self.target.wave_target_feature(width));
        writeln!(
            output,
            "attributes #0 = {{ nounwind \"amdgpu-flat-work-group-size\"=\"{0},{0}\"{wave_attribute}{target_attributes} }}",
            self.flat_workgroup_size()
                .expect("single-kernel emission requires a workgroup size"),
            target_attributes = self.target.llvm_function_attributes()
        )
        .unwrap();
        writeln!(
            output,
            "attributes #1 = {{ nounwind readnone speculatable willreturn }}"
        )
        .unwrap();
        if has_convergent_operation {
            writeln!(output, "attributes #2 = {{ convergent nounwind }}").unwrap();
        }
        writeln!(output).unwrap();
        let workgroup_size = self
            .workgroup_size
            .expect("single-kernel emission requires a workgroup size");
        writeln!(
            output,
            "!0 = !{{i32 {}, i32 {}, i32 {}}}",
            workgroup_size.x, workgroup_size.y, workgroup_size.z
        )
        .unwrap();
        Ok(output)
    }

    fn emit_compiler_module_definition(
        &self,
        output: &mut dyn fmt::Write,
        kernel_attribute: Option<usize>,
        kernel_metadata: Option<usize>,
    ) -> Result<(), LoweringErrors> {
        let parameters = self.llvm_parameters()?.join(", ");
        if self.kernel.is_some() {
            writeln!(
                output,
                "define amdgpu_kernel void @{}({parameters}) #{} !reqd_work_group_size !{} {{",
                self.symbol,
                kernel_attribute.expect("kernel attribute index"),
                kernel_metadata.expect("kernel metadata index"),
            )
            .unwrap();
        } else {
            let result = llvm_result_type(&self.function.signature);
            let wave_attribute = self
                .wave_width
                .map_or("", |width| self.target.wave_target_feature(width));
            let linkage = match self.function.role {
                FunctionRole::InternalHelper => "internal ",
                FunctionRole::DeviceFfiExport => "",
                FunctionRole::KernelEntry | FunctionRole::ExternalImport => {
                    unreachable!("helper definition has a definition role")
                }
            };
            writeln!(
                output,
                "define {linkage}{result} @{}({parameters}) nounwind{wave_attribute}{} {{",
                self.symbol,
                self.target.llvm_function_attributes(),
            )
            .unwrap();
        }
        self.emit_body(output)?;
        writeln!(output, "}}\n").unwrap();
        Ok(())
    }

    fn emit_body(&self, output: &mut dyn fmt::Write) -> Result<(), LoweringErrors> {
        let body = self.function.body.as_ref().expect("definition required");
        for block in &body.blocks {
            writeln!(output, "{}:", block_label(block.id)).unwrap();
            self.emit_block_parameters(output, block);
            for (operation_index, operation) in block.operations.iter().enumerate() {
                self.emit_operation(output, block.id, operation_index, operation)?;
            }
            self.emit_terminator(
                output,
                block.id,
                block.terminator.as_ref().expect("verified terminator"),
            );
            self.emit_split_edges(output, block);
        }
        Ok(())
    }

    fn has_workgroup_barrier(&self) -> bool {
        self.function
            .body
            .iter()
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(&operation.kind, OperationKind::WorkgroupBarrier(_)))
    }

    fn has_wave_kind(&self, predicate: impl Fn(&WaveOperationKind) -> bool) -> bool {
        self.function
            .body
            .iter()
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .any(|operation| {
                matches!(&operation.kind, OperationKind::Wave(wave) if predicate(&wave.kind))
            })
    }

    fn has_matrix_kind(&self, predicate: impl Fn(&MatrixOperationKind) -> bool) -> bool {
        self.function
            .body
            .iter()
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .any(|operation| {
                matches!(&operation.kind, OperationKind::Matrix(matrix) if predicate(&matrix.kind))
            })
    }

    fn has_gfx950_transpose_kind(
        &self,
        predicate: impl Fn(&Gfx950LdsTransposeOperationKindV1) -> bool,
    ) -> bool {
        self.function
            .body
            .iter()
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .any(|operation| {
                matches!(
                    &operation.kind,
                    OperationKind::Gfx950LdsTranspose(transpose)
                        if predicate(&transpose.kind)
                )
            })
    }

    fn emit_workgroup_memory_declarations(&self, output: &mut dyn fmt::Write) -> bool {
        let mut emitted = false;
        let body = self.function.body.as_ref().expect("definition required");
        for operation in body.blocks.iter().flat_map(|block| &block.operations) {
            if !matches!(
                operation.kind,
                OperationKind::WorkgroupMemory(_)
                    | OperationKind::Gfx950LdsTranspose(Gfx950LdsTransposeOperationV1 {
                        kind: Gfx950LdsTransposeOperationKindV1::Current { .. },
                        ..
                    })
            ) {
                continue;
            }
            emitted = true;
            let result = operation.results.first().expect("verified LDS result");
            let symbol = lds_symbol(
                self.kernel
                    .expect("workgroup memory declarations require a kernel"),
                result.id,
            );
            match &operation.kind {
                OperationKind::WorkgroupMemory(memory) => {
                    let element = llvm_type(&memory.element);
                    match memory.extent {
                        WorkgroupMemoryExtent::Static(elements) => writeln!(
                            output,
                            "{symbol} = internal addrspace(3) global [{elements} x {element}] undef, align {}",
                            memory.alignment
                        )
                        .unwrap(),
                        WorkgroupMemoryExtent::Dynamic
                        | WorkgroupMemoryExtent::DynamicAtLeast(_) => writeln!(
                            output,
                            "{symbol} = external addrspace(3) global [0 x {element}], align {}",
                            memory.alignment
                        )
                        .unwrap(),
                    }
                }
                OperationKind::Gfx950LdsTranspose(transpose)
                    if let Gfx950LdsTransposeOperationKindV1::Current { format } =
                        transpose.kind =>
                {
                    let bytes = format.lds_bytes();
                    writeln!(
                        output,
                        "{symbol} = internal addrspace(3) global [{bytes} x i8] undef, align 64"
                    )
                    .unwrap();
                }
                _ => unreachable!("declaration inventory is exact"),
            }
        }
        emitted
    }

    fn emit_block_parameters(&self, output: &mut dyn fmt::Write, block: &BasicBlock) {
        let incomings = self.incoming_edges(block.id);
        for (parameter_index, parameter) in block.parameters.iter().enumerate() {
            match self
                .bindings
                .get(&parameter.id)
                .expect("validated block parameter")
            {
                ValueBinding::Value { llvm_name, ty } => {
                    let values = incomings
                        .iter()
                        .map(|(predecessor, ordinal, arguments)| {
                            let (argument, _) = self.value(arguments[parameter_index]);
                            format!(
                                "[ {argument}, %{} ]",
                                self.phi_predecessor_label(*predecessor, *ordinal, block.id)
                            )
                        })
                        .collect::<Vec<_>>();
                    writeln!(
                        output,
                        "  {llvm_name} = phi {} {}",
                        llvm_type(ty),
                        values.join(", ")
                    )
                    .unwrap();
                }
                ValueBinding::Slice {
                    data_name,
                    length_name,
                    ..
                } => {
                    let data_values = incomings
                        .iter()
                        .map(|(predecessor, ordinal, arguments)| {
                            let ValueBinding::Slice {
                                data_name: argument,
                                ..
                            } = self
                                .bindings
                                .get(&arguments[parameter_index])
                                .expect("verified branch argument")
                            else {
                                unreachable!("verify_module checked branch argument types")
                            };
                            format!(
                                "[ {argument}, %{} ]",
                                self.phi_predecessor_label(*predecessor, *ordinal, block.id)
                            )
                        })
                        .collect::<Vec<_>>();
                    let length_values = incomings
                        .iter()
                        .map(|(predecessor, ordinal, arguments)| {
                            let ValueBinding::Slice {
                                length_name: argument,
                                ..
                            } = self
                                .bindings
                                .get(&arguments[parameter_index])
                                .expect("verified branch argument")
                            else {
                                unreachable!("verify_module checked branch argument types")
                            };
                            format!(
                                "[ {argument}, %{} ]",
                                self.phi_predecessor_label(*predecessor, *ordinal, block.id)
                            )
                        })
                        .collect::<Vec<_>>();
                    writeln!(
                        output,
                        "  {data_name} = phi ptr addrspace(1) {}",
                        data_values.join(", ")
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {length_name} = phi i64 {}",
                        length_values.join(", ")
                    )
                    .unwrap();
                }
            }
        }
    }

    fn llvm_parameters(&self) -> Result<Vec<String>, LoweringErrors> {
        self.function
            .signature
            .parameters
            .iter()
            .enumerate()
            .map(|(index, ty)| match ty {
                Type::Scalar(scalar) => Ok(format!("{} %arg{index}", llvm_scalar(*scalar))),
                Type::Pointer(_) if self.kernel.is_none() => {
                    Ok(format!("{} %arg{index}", llvm_type(ty)))
                }
                Type::Pointer(pointer)
                    if self.kernel.is_some()
                        && pointer.address_space == KernelAddressSpace::Global
                        && supported_memory_type(&pointer.pointee, self.target) =>
                {
                    Ok(format!("{} %arg{index}", llvm_type(ty)))
                }
                Type::Slice(_) => Ok(format!(
                    "ptr addrspace(1) %arg{index}.data, i64 %arg{index}.len"
                )),
                _ => Err(LoweringErrors::one(
                    self.function_location(),
                    LoweringDiagnosticCode::UnsupportedParameter,
                    format!("unsupported kernel parameter {index}: {ty:?}"),
                )),
            })
            .collect()
    }

    fn emit_operation(
        &self,
        output: &mut dyn fmt::Write,
        block: BlockId,
        operation_index: usize,
        operation: &Operation,
    ) -> Result<(), LoweringErrors> {
        let result_name = operation
            .results
            .first()
            .map(|result| value_name(result.id));
        match &operation.kind {
            OperationKind::Constant(_) => {}
            OperationKind::SliceLength { slice } => {
                let ValueBinding::Slice { length_name, .. } =
                    self.bindings.get(slice).expect("validated slice binding")
                else {
                    unreachable!()
                };
                writeln!(
                    output,
                    "  {} = add i64 {}, 0",
                    result_name.expect("validated result"),
                    length_name
                )
                .unwrap();
            }
            OperationKind::SliceData { slice } => {
                let ValueBinding::Slice { data_name, .. } =
                    self.bindings.get(slice).expect("validated slice binding")
                else {
                    unreachable!()
                };
                writeln!(
                    output,
                    "  {} = getelementptr i8, ptr addrspace(1) {}, i64 0",
                    result_name.expect("validated result"),
                    data_name
                )
                .unwrap();
            }
            OperationKind::Intrinsic(intrinsic) => {
                let result = result_name.expect("validated result");
                if matches!(
                    intrinsic.kind,
                    IntrinsicKind::InvocationIndex {
                        kind: IndexKind::Global,
                        axis: Axis::X,
                    }
                ) {
                    self.emit_logical_global_id(output, &result);
                    return Ok(());
                }
                match intrinsic.kind {
                    IntrinsicKind::InvocationIndex {
                        kind: IndexKind::Local,
                        axis: Axis::X,
                    } => {
                        writeln!(
                            output,
                            "  {result}.local.i32 = call i32 @{}()",
                            AmdgcnIntrinsic::WorkItemId(Dim::X).llvm_name()
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  {result}.local = zext i32 {result}.local.i32 to i64"
                        )
                        .unwrap();
                        writeln!(output, "  {result} = add i64 {result}.local, 0").unwrap();
                    }
                    IntrinsicKind::InvocationIndex {
                        kind: IndexKind::Workgroup,
                        axis: Axis::X,
                    } => {
                        writeln!(
                            output,
                            "  {result}.group.i32 = call i32 @{}()",
                            AmdgcnIntrinsic::WorkGroupId(Dim::X).llvm_name()
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  {result}.group = zext i32 {result}.group.i32 to i64"
                        )
                        .unwrap();
                        writeln!(output, "  {result} = add i64 {result}.group, 0").unwrap();
                    }
                    IntrinsicKind::InvocationIndex {
                        kind: IndexKind::WorkgroupSize,
                        axis: Axis::X,
                    } => {
                        let extent = self
                            .workgroup_size
                            .expect("validated workgroup-size intrinsic")
                            .x;
                        writeln!(output, "  {result} = add i64 {extent}, 0").unwrap();
                    }
                    IntrinsicKind::InvocationIndex {
                        kind: IndexKind::WorkgroupCount,
                        axis: Axis::X,
                    } => {
                        let extent = self
                            .workgroup_size
                            .expect("validated workgroup-count intrinsic")
                            .x;
                        writeln!(
                            output,
                            "  {result}.dispatch = call ptr addrspace(4) @{}()",
                            AmdgcnIntrinsic::DispatchPtr.llvm_name()
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  {result}.grid.ptr = getelementptr inbounds i8, ptr addrspace(4) {result}.dispatch, i64 12"
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  {result}.grid.i32 = load i32, ptr addrspace(4) {result}.grid.ptr, align 4"
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  {result}.grid = zext i32 {result}.grid.i32 to i64"
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  {result}.rounded = add i64 {result}.grid, {}",
                            extent - 1
                        )
                        .unwrap();
                        writeln!(output, "  {result} = udiv i64 {result}.rounded, {extent}")
                            .unwrap();
                    }
                    _ => unreachable!("preflight rejected unsupported intrinsic"),
                }
            }
            OperationKind::MemoryIntrinsic(intrinsic) => self.emit_memory_intrinsic(
                output,
                block,
                operation_index,
                result_name.as_deref(),
                intrinsic,
            ),
            OperationKind::Unary { op, operand } => {
                let result = result_name.expect("validated unary result");
                let (operand, ty) = self.value(*operand);
                match unary_lowering_style(*op, ty).expect("preflight accepted the unary operation")
                {
                    LlvmUnaryStyle::FloatNegate => {
                        writeln!(output, "  {result} = fneg {} {operand}", llvm_type(ty)).unwrap();
                    }
                    LlvmUnaryStyle::SignedNegate => {
                        writeln!(output, "  {result} = sub {} 0, {operand}", llvm_type(ty))
                            .unwrap();
                    }
                    LlvmUnaryStyle::BooleanNot => {
                        writeln!(output, "  {result} = xor i1 {operand}, true").unwrap();
                    }
                    LlvmUnaryStyle::IntegerNot => {
                        writeln!(output, "  {result} = xor {} {operand}, -1", llvm_type(ty))
                            .unwrap();
                    }
                }
            }
            OperationKind::Binary { op, lhs, rhs } => {
                if matches!(op, BinaryOp::Checked(_)) {
                    self.emit_checked_binary(output, block, operation_index, operation);
                    return Ok(());
                }
                let (lhs_name, lhs_ty) = self.value(*lhs);
                let (rhs_name, _) = self.value(*rhs);
                writeln!(
                    output,
                    "  {} = {} {} {}, {}",
                    result_name.expect("validated result"),
                    binary_opcode(*op, lhs_ty),
                    llvm_type(lhs_ty),
                    lhs_name,
                    rhs_name
                )
                .unwrap();
            }
            OperationKind::Compare {
                predicate,
                lhs,
                rhs,
            } => {
                let (lhs_name, lhs_ty) = self.value(*lhs);
                let (rhs_name, _) = self.value(*rhs);
                writeln!(
                    output,
                    "  {} = {} {} {} {}, {}",
                    result_name.expect("validated result"),
                    compare_opcode(lhs_ty),
                    compare_predicate(*predicate, lhs_ty),
                    llvm_type(lhs_ty),
                    lhs_name,
                    rhs_name
                )
                .unwrap();
            }
            OperationKind::Select {
                condition,
                true_value,
                false_value,
            } => {
                let (condition_name, _) = self.value(*condition);
                let (true_name, value_ty) = self.value(*true_value);
                let (false_name, _) = self.value(*false_value);
                writeln!(
                    output,
                    "  {} = select i1 {}, {} {}, {} {}",
                    result_name.expect("validated result"),
                    condition_name,
                    llvm_type(value_ty),
                    true_name,
                    llvm_type(value_ty),
                    false_name
                )
                .unwrap();
            }
            OperationKind::Cast { kind, value, to } => {
                let (value_name, from) = self.value(*value);
                if *kind == CastKind::Bitcast && llvm_type(from) == llvm_type(to) {
                    writeln!(
                        output,
                        "  {} = add {} {}, 0",
                        result_name.expect("validated result"),
                        llvm_type(from),
                        value_name
                    )
                    .unwrap();
                    return Ok(());
                }
                writeln!(
                    output,
                    "  {} = {} {} {} to {}",
                    result_name.expect("validated result"),
                    cast_opcode(*kind, from, to),
                    llvm_type(from),
                    value_name,
                    llvm_type(to)
                )
                .unwrap();
            }
            OperationKind::GetElementPointer { base, offset } => {
                let (base_name, base_ty) = self.value(*base);
                let (offset_name, offset_ty) = self.value(*offset);
                let Type::Pointer(pointer) = base_ty else {
                    unreachable!()
                };
                let address_space = llvm_address_space(pointer.address_space);
                writeln!(
                    output,
                    "  {} = getelementptr {}, ptr addrspace({}) {}, {} {}",
                    result_name.expect("validated result"),
                    llvm_type(&pointer.pointee),
                    address_space,
                    base_name,
                    llvm_type(offset_ty),
                    offset_name
                )
                .unwrap();
            }
            OperationKind::Load { pointer, access } => {
                let (pointer_name, pointer_ty) = self.value(*pointer);
                let Type::Pointer(pointer_ty) = pointer_ty else {
                    unreachable!()
                };
                let address_space = llvm_address_space(pointer_ty.address_space);
                let volatile = if access.volatile { " volatile" } else { "" };
                writeln!(
                    output,
                    "  {} = load{} {}, ptr addrspace({}) {}, align {}",
                    result_name.expect("validated result"),
                    volatile,
                    llvm_type(&pointer_ty.pointee),
                    address_space,
                    pointer_name,
                    access.alignment
                )
                .unwrap();
            }
            OperationKind::GuardedLoad {
                pointer,
                predicate,
                fallback,
                access,
            } => {
                let result = result_name.expect("validated guarded-load result");
                let (pointer_name, pointer_ty) = self.value(*pointer);
                let (predicate_name, _) = self.value(*predicate);
                let (fallback_name, fallback_ty) = self.value(*fallback);
                let Type::Pointer(pointer_ty) = pointer_ty else {
                    unreachable!()
                };
                let address_space = llvm_address_space(pointer_ty.address_space);
                let volatile = if access.volatile { " volatile" } else { "" };
                let true_label = guarded_load_true_label(block, operation_index);
                let false_label = guarded_load_false_label(block, operation_index);
                let merge_label = guarded_load_merge_label(block, operation_index);
                writeln!(
                    output,
                    "  br i1 {predicate_name}, label %{true_label}, label %{false_label}"
                )
                .unwrap();
                writeln!(output, "{true_label}:").unwrap();
                writeln!(
                    output,
                    "  {result}.loaded = load{volatile} {}, ptr addrspace({address_space}) {pointer_name}, align {}",
                    llvm_type(&pointer_ty.pointee),
                    access.alignment,
                )
                .unwrap();
                writeln!(output, "  br label %{merge_label}").unwrap();
                writeln!(output, "{false_label}:").unwrap();
                writeln!(output, "  br label %{merge_label}").unwrap();
                writeln!(output, "{merge_label}:").unwrap();
                writeln!(
                    output,
                    "  {result} = phi {} [ {result}.loaded, %{true_label} ], [ {fallback_name}, %{false_label} ]",
                    llvm_type(fallback_ty),
                )
                .unwrap();
            }
            OperationKind::Store {
                pointer,
                value,
                access,
            } => {
                let (pointer_name, pointer_ty) = self.value(*pointer);
                let (value_name, _) = self.value(*value);
                let Type::Pointer(pointer_ty) = pointer_ty else {
                    unreachable!()
                };
                let address_space = llvm_address_space(pointer_ty.address_space);
                let volatile = if access.volatile { " volatile" } else { "" };
                writeln!(
                    output,
                    "  store{} {} {}, ptr addrspace({}) {}, align {}",
                    volatile,
                    llvm_type(&pointer_ty.pointee),
                    value_name,
                    address_space,
                    pointer_name,
                    access.alignment
                )
                .unwrap();
            }
            OperationKind::Alloca {
                element,
                count: None,
                address_space: KernelAddressSpace::Private,
                alignment,
            } => {
                writeln!(
                    output,
                    "  {} = alloca {}, align {}, addrspace(5)",
                    result_name.expect("validated private allocation result"),
                    llvm_type(element),
                    alignment
                )
                .unwrap();
            }
            OperationKind::Call { callee, arguments } => {
                if let Some(diagnostic) =
                    AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
                {
                    self.emit_diagnostic(output, result_name.as_deref(), &diagnostic);
                    return Ok(());
                }
                if let Some(float) = FloatOperation::from_intrinsic_call(callee, arguments) {
                    self.emit_float(
                        output,
                        result_name.as_deref().expect("verified float result"),
                        &float,
                    );
                    return Ok(());
                }
                let callee_function = self
                    .module
                    .function(callee)
                    .expect("verify_module checked the callee");
                let symbol = self
                    .call_symbols
                    .expect("compiler-module call emission requires a symbol table")
                    .get(callee)
                    .expect("compiler-module preflight rejected kernel-entry calls");
                let arguments = arguments
                    .iter()
                    .map(|argument| {
                        let (name, ty) = self.value(*argument);
                        format!("{} {name}", llvm_type(ty))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                match callee_function.signature.results.as_slice() {
                    [] => writeln!(output, "  call void @{symbol}({arguments})").unwrap(),
                    [result] => writeln!(
                        output,
                        "  {} = call {} @{symbol}({arguments})",
                        result_name.expect("verified call result"),
                        llvm_type(result)
                    )
                    .unwrap(),
                    _ => unreachable!("compiler-module preflight rejected multi-value returns"),
                }
            }
            OperationKind::Atomic(atomic) => {
                self.emit_atomic(output, operation, atomic);
            }
            OperationKind::Fence(fence) => {
                emit_fence(output, fence.memory_scope, fence.semantics.ordering);
            }
            OperationKind::WorkgroupBarrier(barrier) => {
                match barrier.semantics.ordering {
                    MemoryOrdering::Acquire => {}
                    MemoryOrdering::Release | MemoryOrdering::AcquireRelease => {
                        emit_fence(output, barrier.memory_scope, MemoryOrdering::Release);
                    }
                    MemoryOrdering::SequentiallyConsistent => {
                        emit_fence(
                            output,
                            barrier.memory_scope,
                            MemoryOrdering::SequentiallyConsistent,
                        );
                    }
                    MemoryOrdering::Relaxed => {
                        unreachable!("kernel IR verification rejected a relaxed barrier")
                    }
                }
                if self.target.requires_physical_workgroup_barrier() {
                    // The gfx942 target policy retains the source barrier as a
                    // physical instruction even when LLVM can prove one wave.
                    writeln!(output, "  call void asm sideeffect \"s_barrier\", \"\"()").unwrap();
                } else {
                    writeln!(
                        output,
                        "  call void @{}()",
                        AmdgcnIntrinsic::SBarrier.llvm_name()
                    )
                    .unwrap();
                }
                match barrier.semantics.ordering {
                    MemoryOrdering::Release => {}
                    MemoryOrdering::Acquire | MemoryOrdering::AcquireRelease => {
                        emit_fence(output, barrier.memory_scope, MemoryOrdering::Acquire);
                    }
                    MemoryOrdering::SequentiallyConsistent => {
                        emit_fence(
                            output,
                            barrier.memory_scope,
                            MemoryOrdering::SequentiallyConsistent,
                        );
                    }
                    MemoryOrdering::Relaxed => unreachable!(),
                }
            }
            OperationKind::WorkgroupMemory(memory) => {
                let result = operation.results.first().expect("verified LDS result");
                let result_name = result_name.expect("verified LDS result name");
                let elements = match memory.extent {
                    WorkgroupMemoryExtent::Static(elements) => elements,
                    WorkgroupMemoryExtent::Dynamic | WorkgroupMemoryExtent::DynamicAtLeast(_) => 0,
                };
                let element = llvm_type(&memory.element);
                writeln!(
                    output,
                    "  {result_name} = getelementptr [{elements} x {element}], ptr addrspace(3) {}, i32 0, i32 0",
                    lds_symbol(
                        self.kernel
                            .expect("workgroup memory emission requires a kernel"),
                        result.id,
                    )
                )
                .unwrap();
            }
            OperationKind::InlineAssembly(assembly) => {
                self.emit_inline_assembly(output, operation, assembly);
            }
            OperationKind::Matrix(matrix) => {
                self.emit_matrix(output, block, operation_index, operation, matrix);
            }
            OperationKind::Gfx950LdsTranspose(transpose) => {
                self.emit_gfx950_lds_transpose(output, operation, transpose);
            }
            OperationKind::Wave(wave) => {
                self.emit_wave(
                    output,
                    result_name.as_deref().expect("verified wave result"),
                    wave,
                );
            }
            _ => unreachable!("preflight rejected unsupported operation"),
        }
        Ok(())
    }

    fn emit_checked_binary(
        &self,
        output: &mut dyn fmt::Write,
        block: BlockId,
        operation_index: usize,
        operation: &Operation,
    ) {
        let OperationKind::Binary {
            op: BinaryOp::Checked(operator),
            lhs,
            rhs,
        } = &operation.kind
        else {
            unreachable!("checked binary emitter requires a checked binary operation")
        };
        let (lhs_name, lhs_ty) = self.value(*lhs);
        let (rhs_name, _) = self.value(*rhs);
        let scalar = lhs_ty
            .as_scalar()
            .expect("KIR verification requires checked integer operands");
        let ty = llvm_scalar(scalar);
        let intrinsic = checked_binary_intrinsic_name(*operator, scalar);
        let pair = format!("checked.{}.{}", block.0, operation_index);
        let value = value_name(operation.results[0].id);
        let overflow = value_name(operation.results[1].id);
        writeln!(
            output,
            "  %{pair} = call {{ {ty}, i1 }} @{intrinsic}({ty} {lhs_name}, {ty} {rhs_name})"
        )
        .unwrap();
        writeln!(output, "  {value} = extractvalue {{ {ty}, i1 }} %{pair}, 0").unwrap();
        writeln!(
            output,
            "  {overflow} = extractvalue {{ {ty}, i1 }} %{pair}, 1"
        )
        .unwrap();
    }

    fn emit_matrix(
        &self,
        output: &mut dyn fmt::Write,
        block: BlockId,
        operation_index: usize,
        operation: &Operation,
        matrix: &MatrixOperation,
    ) {
        let temporary = format!("matrix.{}.{}", block.0, operation_index);
        match &matrix.kind {
            MatrixOperationKind::MultiplyAccumulate {
                lhs,
                rhs,
                accumulator,
                ..
            } => {
                for (label, values, ty) in [
                    ("lhs", lhs.as_slice(), "i16"),
                    ("rhs", rhs.as_slice(), "i16"),
                    ("acc", accumulator.as_slice(), "float"),
                ] {
                    for (index, value) in values.iter().enumerate() {
                        let source = self.value(*value).0;
                        let prior = if index == 0 {
                            "poison".to_string()
                        } else {
                            format!("%{temporary}.{label}.{}", index - 1)
                        };
                        writeln!(
                            output,
                            "  %{temporary}.{label}.{index} = insertelement <4 x {ty}> {prior}, {ty} {source}, i64 {index}"
                        )
                        .unwrap();
                    }
                }
                writeln!(
                    output,
                    "  %{temporary}.mfma = call <4 x float> @{}(<4 x i16> %{temporary}.lhs.3, <4 x i16> %{temporary}.rhs.3, <4 x float> %{temporary}.acc.3, i32 0, i32 0, i32 0)",
                    AmdgcnIntrinsic::MfmaF32M16N16K16Bf16.llvm_name()
                )
                .unwrap();
                for (index, result) in operation.results.iter().enumerate() {
                    writeln!(
                        output,
                        "  {} = extractelement <4 x float> %{temporary}.mfma, i64 {index}",
                        value_name(result.id)
                    )
                    .unwrap();
                }
            }
            MatrixOperationKind::ScaledMultiplyAccumulate {
                lhs,
                rhs,
                accumulator,
                profile,
            } => {
                for (label, values, ty) in [
                    ("lhs", lhs.as_slice(), "i32"),
                    ("rhs", rhs.as_slice(), "i32"),
                    ("acc", accumulator.as_slice(), "float"),
                ] {
                    let width = values.len();
                    for (index, value) in values.iter().enumerate() {
                        let source = self.value(*value).0;
                        let prior = if index == 0 {
                            "poison".to_string()
                        } else {
                            format!("%{temporary}.{label}.{}", index - 1)
                        };
                        writeln!(
                            output,
                            "  %{temporary}.{label}.{index} = insertelement <{width} x {ty}> {prior}, {ty} {source}, i64 {index}"
                        )
                        .unwrap();
                    }
                }
                let mixed_fp4_by_fp8 = matrix.tensor_layout.as_ref().is_some_and(|contract| {
                    contract.profile
                        == TensorInstructionProfileV1::Gfx950ScaledMfmaFp4E2M1Fp8E4M3F32M16N16K128Wave64
                });
                let lhs_format_selector =
                    if *profile == MatrixMultiplyProfile::fp4_e2m1_f32_m16n16k128_wave64() {
                        4
                    } else {
                        0
                    };
                let rhs_format_selector = if mixed_fp4_by_fp8 {
                    0
                } else {
                    lhs_format_selector
                };
                writeln!(
                    output,
                    "  %{temporary}.mfma = call <4 x float> @{}(<8 x i32> %{temporary}.lhs.7, <8 x i32> %{temporary}.rhs.7, <4 x float> %{temporary}.acc.3, i32 {lhs_format_selector}, i32 {rhs_format_selector}, i32 0, i32 0, i32 0, i32 0)",
                    AmdgcnIntrinsic::MfmaScaleF32M16N16K128F8F6F4V8I32.llvm_name(),
                )
                .unwrap();
                for (index, result) in operation.results.iter().enumerate() {
                    writeln!(
                        output,
                        "  {} = extractelement <4 x float> %{temporary}.mfma, i64 {index}",
                        value_name(result.id)
                    )
                    .unwrap();
                }
            }
            MatrixOperationKind::LdsLoad { base, profile } => {
                self.emit_matrix_lds_lane_address(output, &temporary);
                let base = self.value(*base).0;
                let element = llvm_type(&profile.element.ty());
                let alignment = match profile.element {
                    MatrixElement::Bf16 => 2,
                    MatrixElement::F32 => 4,
                    MatrixElement::Fp4E2M1 | MatrixElement::Fp8E4M3 => 1,
                };
                for (index, result) in operation.results.iter().enumerate() {
                    self.emit_matrix_lds_pointer(output, &temporary, index, element, base);
                    writeln!(
                        output,
                        "  {} = load {element}, ptr addrspace(3) %{temporary}.pointer.{index}, align {alignment}",
                        value_name(result.id)
                    )
                    .unwrap();
                }
            }
            MatrixOperationKind::LdsStore {
                base,
                values,
                profile,
            } => {
                self.emit_matrix_lds_lane_address(output, &temporary);
                let base = self.value(*base).0;
                let element = llvm_type(&profile.element.ty());
                let alignment = match profile.element {
                    MatrixElement::Bf16 => 2,
                    MatrixElement::F32 => 4,
                    MatrixElement::Fp4E2M1 | MatrixElement::Fp8E4M3 => 1,
                };
                for (index, value) in values.iter().enumerate() {
                    self.emit_matrix_lds_pointer(output, &temporary, index, element, base);
                    writeln!(
                        output,
                        "  store {element} {}, ptr addrspace(3) %{temporary}.pointer.{index}, align {alignment}",
                        self.value(*value).0
                    )
                    .unwrap();
                }
            }
        }
    }

    fn emit_matrix_lds_lane_address(&self, output: &mut dyn fmt::Write, temporary: &str) {
        writeln!(
            output,
            "  %{temporary}.lane.lo = call i32 @{}(i32 -1, i32 0)",
            AmdgcnIntrinsic::MbcntLo.llvm_name()
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.lane = call i32 @{}(i32 -1, i32 %{temporary}.lane.lo)",
            AmdgcnIntrinsic::MbcntHi.llvm_name()
        )
        .unwrap();
        writeln!(output, "  %{temporary}.row = and i32 %{temporary}.lane, 15").unwrap();
        writeln!(
            output,
            "  %{temporary}.chunk = lshr i32 %{temporary}.lane, 4"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.column.base = shl i32 %{temporary}.chunk, 2"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.swizzle.row = and i32 %{temporary}.row, 3"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.swizzle = shl i32 %{temporary}.swizzle.row, 2"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.row.base = shl i32 %{temporary}.row, 4"
        )
        .unwrap();
    }

    fn emit_matrix_lds_pointer(
        &self,
        output: &mut dyn fmt::Write,
        temporary: &str,
        index: usize,
        element: &str,
        base: &str,
    ) {
        writeln!(
            output,
            "  %{temporary}.column.{index} = add i32 %{temporary}.column.base, {index}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.physical.column.{index} = xor i32 %{temporary}.column.{index}, %{temporary}.swizzle"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.index.{index} = add i32 %{temporary}.row.base, %{temporary}.physical.column.{index}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{temporary}.pointer.{index} = getelementptr {element}, ptr addrspace(3) {base}, i32 %{temporary}.index.{index}"
        )
        .unwrap();
    }

    fn emit_memory_intrinsic(
        &self,
        output: &mut dyn fmt::Write,
        block: BlockId,
        operation_index: usize,
        result_name: Option<&str>,
        intrinsic: &MemoryIntrinsicOperation,
    ) {
        let temporary = format!("memory.{}.{}", block.0, operation_index);
        match *intrinsic {
            MemoryIntrinsicOperation::PointerDistance {
                pointer,
                origin,
                kind,
                unit,
                layout,
                address_space,
                contract,
                ..
            } => {
                debug_assert_eq!(contract, PointerDistanceContract::supported_rust(kind));
                let result = result_name.expect("verified pointer-distance result");
                let (pointer_name, _) = self.value(pointer);
                let (origin_name, _) = self.value(origin);
                let address_space = llvm_address_space(address_space);
                writeln!(
                    output,
                    "  %{temporary}.pointer = ptrtoint ptr addrspace({address_space}) {pointer_name} to i64"
                )
                .unwrap();
                writeln!(
                    output,
                    "  %{temporary}.origin = ptrtoint ptr addrspace({address_space}) {origin_name} to i64"
                )
                .unwrap();
                let subtraction = match kind {
                    PointerDistanceKind::Signed => "sub",
                    PointerDistanceKind::Unsigned => "sub nuw",
                };
                writeln!(
                    output,
                    "  %{temporary}.bytes = {subtraction} i64 %{temporary}.pointer, %{temporary}.origin"
                )
                .unwrap();
                let divisor = match unit {
                    PointerDistanceUnit::Elements => layout.size_bytes,
                    PointerDistanceUnit::Bytes => 1,
                };
                let division = match kind {
                    PointerDistanceKind::Signed => "sdiv exact",
                    PointerDistanceKind::Unsigned => "udiv exact",
                };
                writeln!(
                    output,
                    "  {result} = {division} i64 %{temporary}.bytes, {divisor}"
                )
                .unwrap();
            }
            MemoryIntrinsicOperation::VolatileLoad {
                pointer,
                element,
                address_space,
                layout,
                contract,
            } => {
                debug_assert!(contract.matches_supported_load(element, address_space));
                if element == MemoryElementType::Unit {
                    return;
                }
                let (pointer_name, _) = self.value(pointer);
                let element_type = element.ir_type();
                writeln!(
                    output,
                    "  {} = load volatile {}, ptr addrspace({}) {}, align {}",
                    result_name.expect("verified volatile-load result"),
                    llvm_type(&element_type),
                    llvm_address_space(address_space),
                    pointer_name,
                    layout.alignment_bytes
                )
                .unwrap();
            }
            MemoryIntrinsicOperation::VolatileStore {
                pointer,
                value,
                element,
                address_space,
                layout,
                contract,
            } => {
                debug_assert!(contract.matches_supported_store(element, address_space));
                if element == MemoryElementType::Unit {
                    return;
                }
                let (pointer_name, _) = self.value(pointer);
                let (value_name, _) = self.value(value);
                let element_type = element.ir_type();
                writeln!(
                    output,
                    "  store volatile {} {}, ptr addrspace({}) {}, align {}",
                    llvm_type(&element_type),
                    value_name,
                    llvm_address_space(address_space),
                    pointer_name,
                    layout.alignment_bytes
                )
                .unwrap();
            }
            MemoryIntrinsicOperation::CopyNonOverlapping {
                source,
                destination,
                count,
                element,
                source_address_space,
                destination_address_space,
                layout,
                contract,
                ..
            } => {
                debug_assert_eq!(
                    contract,
                    fe2o3_kernel_ir::CopyNonOverlappingContract::supported_rust()
                );
                if element == MemoryElementType::Unit {
                    return;
                }
                let (source_name, _) = self.value(source);
                let (destination_name, _) = self.value(destination);
                let (count_name, _) = self.value(count);
                let source_address_space = llvm_address_space(source_address_space);
                let destination_address_space = llvm_address_space(destination_address_space);
                writeln!(
                    output,
                    "  %{temporary}.bytes = mul nuw i64 {count_name}, {}",
                    layout.size_bytes
                )
                .unwrap();
                writeln!(
                    output,
                    "  call void @llvm.memcpy.p{destination_address_space}.p{source_address_space}.i64(ptr addrspace({destination_address_space}) align {} {destination_name}, ptr addrspace({source_address_space}) align {} {source_name}, i64 %{temporary}.bytes, i1 false)",
                    layout.alignment_bytes,
                    layout.alignment_bytes,
                )
                .unwrap();
            }
        }
    }

    fn emit_inline_assembly(
        &self,
        output: &mut dyn fmt::Write,
        operation: &Operation,
        assembly: &InlineAssembly,
    ) {
        let instruction = gfx942_assembly_instruction(&assembly.mnemonic)
            .expect("preflight admitted a closed gfx942 instruction");
        let result = operation
            .results
            .first()
            .expect("preflight required one assembly result");
        let placeholders = (0..assembly.operands.len())
            .map(|index| format!("${index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let register = match instruction.constraint {
            AssemblyConstraint::Sgpr32 => "s",
            AssemblyConstraint::Vgpr32 => "v",
            AssemblyConstraint::ImmediateI32 => {
                unreachable!("the bounded gfx942 subset has no immediate outputs")
            }
        };
        let constraints = std::iter::once(format!("={register}"))
            .chain((0..instruction.input_count).map(|_| register.to_owned()))
            .collect::<Vec<_>>()
            .join(",");
        let inputs = assembly.operands[1..]
            .iter()
            .map(|operand| {
                let AssemblyOperandKind::Input(value) = operand.kind else {
                    unreachable!("preflight required SSA input roles")
                };
                let (name, ty) = self.value(value);
                format!("{} {name}", llvm_type(ty))
            })
            .collect::<Vec<_>>()
            .join(", ");
        let side_effect = if assembly.options.contains(&AssemblyOption::Pure) {
            ""
        } else {
            " sideeffect"
        };
        writeln!(
            output,
            "  {} = call {} asm{} \"{} {}\", \"{}\"({})",
            value_name(result.id),
            llvm_type(&result.ty),
            side_effect,
            instruction.mnemonic,
            placeholders,
            constraints,
            inputs
        )
        .unwrap();
    }

    fn emit_gfx950_lds_transpose(
        &self,
        output: &mut dyn fmt::Write,
        operation: &Operation,
        transpose: &Gfx950LdsTransposeOperationV1,
    ) {
        match transpose.kind {
            Gfx950LdsTransposeOperationKindV1::Current { format } => {
                let result = operation
                    .results
                    .first()
                    .expect("verified transpose result");
                let bytes = format.lds_bytes();
                writeln!(
                    output,
                    "  {} = getelementptr [{bytes} x i8], ptr addrspace(3) {}, i32 0, i32 0",
                    value_name(result.id),
                    lds_symbol(
                        self.kernel
                            .expect("gfx950 transpose storage requires a kernel"),
                        result.id,
                    )
                )
                .unwrap();
            }
            Gfx950LdsTransposeOperationKindV1::Stage {
                format,
                storage,
                source_slice,
                offset,
                rows,
                columns,
                stride,
                token_base,
                reduction_base,
            } => self.emit_gfx950_lds_transpose_stage(
                output,
                operation,
                format,
                storage,
                source_slice,
                offset,
                rows,
                columns,
                stride,
                token_base,
                reduction_base,
            ),
            Gfx950LdsTransposeOperationKindV1::Publish { storage, .. } => {
                emit_fence(
                    output,
                    SynchronizationScope::Workgroup,
                    MemoryOrdering::Release,
                );
                writeln!(output, "  call void asm sideeffect \"s_barrier\", \"\"()").unwrap();
                emit_fence(
                    output,
                    SynchronizationScope::Workgroup,
                    MemoryOrdering::Acquire,
                );
                let result = operation
                    .results
                    .first()
                    .expect("verified transpose result");
                writeln!(
                    output,
                    "  {} = getelementptr i8, ptr addrspace(3) {}, i32 0",
                    value_name(result.id),
                    self.value(storage).0,
                )
                .unwrap();
            }
            Gfx950LdsTransposeOperationKindV1::Read { format, storage } => {
                self.emit_gfx950_lds_transpose_read(output, operation, format, storage);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_gfx950_lds_transpose_stage(
        &self,
        output: &mut dyn fmt::Write,
        operation: &Operation,
        format: Gfx950LdsTransposeFormatV1,
        storage: ValueId,
        source_slice: ValueId,
        offset: ValueId,
        rows: ValueId,
        columns: ValueId,
        stride: ValueId,
        token_base: ValueId,
        reduction_base: ValueId,
    ) {
        let result = operation
            .results
            .first()
            .expect("verified transpose result");
        let prefix = format!("v{}.transpose", result.id.0);
        let storage = self.value(storage).0;
        let (source_data, source_length) = match self
            .bindings
            .get(&source_slice)
            .expect("verified transpose source")
        {
            ValueBinding::Slice {
                data_name,
                length_name,
                ..
            } => (data_name.as_str(), length_name.as_str()),
            ValueBinding::Value { .. } => unreachable!("verified transpose source is a slice"),
        };
        let offset = self.value(offset).0;
        let rows = self.value(rows).0;
        let columns = self.value(columns).0;
        let stride = self.value(stride).0;
        let token_base = self.value(token_base).0;
        let reduction_base = self.value(reduction_base).0;

        self.emit_lane_id(output, &format!("%{prefix}.lane"), WaveWidth::Wave64);
        let lane_stride = format.lane_byte_stride();
        writeln!(
            output,
            "  %{prefix}.lds.lane.base = mul i32 %{prefix}.lane, {lane_stride}"
        )
        .unwrap();

        match format {
            Gfx950LdsTransposeFormatV1::Fp4E2M1 => {
                // Match the gfx950 B4 source layout: each contiguous group of
                // sixteen lanes owns 32 reduction elements, split over two
                // eight-byte transpose sources per lane.
                writeln!(
                    output,
                    "  %{prefix}.lane.in.group = and i32 %{prefix}.lane, 15"
                )
                .unwrap();
                writeln!(output, "  %{prefix}.group = lshr i32 %{prefix}.lane, 4").unwrap();
                writeln!(
                    output,
                    "  %{prefix}.group.depth = shl i32 %{prefix}.group, 5"
                )
                .unwrap();
                writeln!(
                    output,
                    "  %{prefix}.depth.base = add i32 %{prefix}.group.depth, %{prefix}.lane.in.group"
                )
                .unwrap();
                for part in 0..2u32 {
                    self.emit_gfx950_stage_column(
                        output,
                        &prefix,
                        part,
                        reduction_base,
                        columns,
                        part * 16,
                    );
                    for byte in 0..8u32 {
                        let low_ordinal = part * 16 + byte * 2;
                        let high_ordinal = low_ordinal + 1;
                        self.emit_gfx950_stage_checked_load(
                            output,
                            &prefix,
                            low_ordinal,
                            part,
                            byte * 2,
                            source_data,
                            source_length,
                            offset,
                            rows,
                            stride,
                            token_base,
                        );
                        self.emit_gfx950_stage_checked_load(
                            output,
                            &prefix,
                            high_ordinal,
                            part,
                            byte * 2 + 1,
                            source_data,
                            source_length,
                            offset,
                            rows,
                            stride,
                            token_base,
                        );
                        writeln!(
                            output,
                            "  %{prefix}.packed.low.{part}.{byte} = and i8 %{prefix}.loaded.{low_ordinal}, 15"
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  %{prefix}.packed.high.mask.{part}.{byte} = and i8 %{prefix}.loaded.{high_ordinal}, 15"
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  %{prefix}.packed.high.{part}.{byte} = shl i8 %{prefix}.packed.high.mask.{part}.{byte}, 4"
                        )
                        .unwrap();
                        writeln!(
                            output,
                            "  %{prefix}.packed.{part}.{byte} = or i8 %{prefix}.packed.low.{part}.{byte}, %{prefix}.packed.high.{part}.{byte}"
                        )
                        .unwrap();
                        self.emit_gfx950_stage_store(
                            output,
                            &prefix,
                            storage,
                            part,
                            byte,
                            &format!("%{prefix}.packed.{part}.{byte}"),
                        );
                    }
                }
            }
            Gfx950LdsTransposeFormatV1::Fp8E4M3 => {
                // B8 divides each sixteen-lane group into two eight-token
                // source bands. Four per-lane parts cover both 64-element K
                // halves consumed by the scaled MFMA instruction.
                writeln!(
                    output,
                    "  %{prefix}.lane.in.group = and i32 %{prefix}.lane, 15"
                )
                .unwrap();
                writeln!(
                    output,
                    "  %{prefix}.token.band.bit = and i32 %{prefix}.lane.in.group, 1"
                )
                .unwrap();
                writeln!(
                    output,
                    "  %{prefix}.token.band = shl i32 %{prefix}.token.band.bit, 3"
                )
                .unwrap();
                writeln!(
                    output,
                    "  %{prefix}.lane.depth = lshr i32 %{prefix}.lane.in.group, 1"
                )
                .unwrap();
                writeln!(output, "  %{prefix}.group = lshr i32 %{prefix}.lane, 4").unwrap();
                writeln!(
                    output,
                    "  %{prefix}.group.depth = shl i32 %{prefix}.group, 4"
                )
                .unwrap();
                writeln!(
                    output,
                    "  %{prefix}.depth.base = add i32 %{prefix}.group.depth, %{prefix}.lane.depth"
                )
                .unwrap();
                for part in 0..4u32 {
                    let depth_offset = (part % 2) * 8 + if part >= 2 { 64 } else { 0 };
                    self.emit_gfx950_stage_column(
                        output,
                        &prefix,
                        part,
                        reduction_base,
                        columns,
                        depth_offset,
                    );
                    for byte in 0..8u32 {
                        let ordinal = part * 8 + byte;
                        self.emit_gfx950_stage_checked_load_with_row_offset(
                            output,
                            &prefix,
                            ordinal,
                            part,
                            byte,
                            &format!("%{prefix}.token.band"),
                            source_data,
                            source_length,
                            offset,
                            rows,
                            stride,
                            token_base,
                        );
                        self.emit_gfx950_stage_store(
                            output,
                            &prefix,
                            storage,
                            part,
                            byte,
                            &format!("%{prefix}.loaded.{ordinal}"),
                        );
                    }
                }
            }
        }
        writeln!(
            output,
            "  {} = getelementptr i8, ptr addrspace(3) {storage}, i32 0",
            value_name(result.id)
        )
        .unwrap();
    }

    fn emit_gfx950_stage_column(
        &self,
        output: &mut dyn fmt::Write,
        prefix: &str,
        part: u32,
        reduction_base: &str,
        columns: &str,
        depth_offset: u32,
    ) {
        writeln!(
            output,
            "  %{prefix}.depth.i32.{part} = add i32 %{prefix}.depth.base, {depth_offset}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.depth.{part} = zext i32 %{prefix}.depth.i32.{part} to i64"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.column.add.{part} = call {{ i64, i1 }} @llvm.uadd.with.overflow.i64(i64 {reduction_base}, i64 %{prefix}.depth.{part})"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.column.{part} = extractvalue {{ i64, i1 }} %{prefix}.column.add.{part}, 0"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.column.overflow.{part} = extractvalue {{ i64, i1 }} %{prefix}.column.add.{part}, 1"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.column.nooverflow.{part} = xor i1 %{prefix}.column.overflow.{part}, true"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.column.inbounds.{part} = icmp ult i64 %{prefix}.column.{part}, {columns}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.column.valid.{part} = and i1 %{prefix}.column.nooverflow.{part}, %{prefix}.column.inbounds.{part}"
        )
        .unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_gfx950_stage_checked_load(
        &self,
        output: &mut dyn fmt::Write,
        prefix: &str,
        ordinal: u32,
        part: u32,
        row_delta: u32,
        source_data: &str,
        source_length: &str,
        offset: &str,
        rows: &str,
        stride: &str,
        token_base: &str,
    ) {
        self.emit_gfx950_stage_checked_load_with_row_offset(
            output,
            prefix,
            ordinal,
            part,
            row_delta,
            "0",
            source_data,
            source_length,
            offset,
            rows,
            stride,
            token_base,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_gfx950_stage_checked_load_with_row_offset(
        &self,
        output: &mut dyn fmt::Write,
        prefix: &str,
        ordinal: u32,
        part: u32,
        row_delta: u32,
        row_offset: &str,
        source_data: &str,
        source_length: &str,
        offset: &str,
        rows: &str,
        stride: &str,
        token_base: &str,
    ) {
        writeln!(
            output,
            "  %{prefix}.row.band.i32.{ordinal} = add i32 {row_offset}, {row_delta}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.band.{ordinal} = zext i32 %{prefix}.row.band.i32.{ordinal} to i64"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.add.{ordinal} = call {{ i64, i1 }} @llvm.uadd.with.overflow.i64(i64 {token_base}, i64 %{prefix}.row.band.{ordinal})"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.row.add.{ordinal}, 0"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.overflow.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.row.add.{ordinal}, 1"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.nooverflow.{ordinal} = xor i1 %{prefix}.row.overflow.{ordinal}, true"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.inbounds.{ordinal} = icmp ult i64 %{prefix}.row.{ordinal}, {rows}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.valid.0.{ordinal} = and i1 %{prefix}.row.nooverflow.{ordinal}, %{prefix}.row.inbounds.{ordinal}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.mul.{ordinal} = call {{ i64, i1 }} @llvm.umul.with.overflow.i64(i64 %{prefix}.row.{ordinal}, i64 {stride})"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.offset.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.row.mul.{ordinal}, 0"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.mul.overflow.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.row.mul.{ordinal}, 1"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.row.mul.nooverflow.{ordinal} = xor i1 %{prefix}.row.mul.overflow.{ordinal}, true"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.linear.add.{ordinal} = call {{ i64, i1 }} @llvm.uadd.with.overflow.i64(i64 %{prefix}.row.offset.{ordinal}, i64 %{prefix}.column.{part})"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.linear.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.linear.add.{ordinal}, 0"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.linear.overflow.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.linear.add.{ordinal}, 1"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.linear.nooverflow.{ordinal} = xor i1 %{prefix}.linear.overflow.{ordinal}, true"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.index.add.{ordinal} = call {{ i64, i1 }} @llvm.uadd.with.overflow.i64(i64 {offset}, i64 %{prefix}.linear.{ordinal})"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.index.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.index.add.{ordinal}, 0"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.index.overflow.{ordinal} = extractvalue {{ i64, i1 }} %{prefix}.index.add.{ordinal}, 1"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.index.nooverflow.{ordinal} = xor i1 %{prefix}.index.overflow.{ordinal}, true"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.index.inbounds.{ordinal} = icmp ult i64 %{prefix}.index.{ordinal}, {source_length}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.guard.0.{ordinal} = and i1 %{prefix}.row.valid.0.{ordinal}, %{prefix}.column.valid.{part}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.guard.0a.{ordinal} = and i1 %{prefix}.guard.0.{ordinal}, %{prefix}.row.mul.nooverflow.{ordinal}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.guard.1.{ordinal} = and i1 %{prefix}.guard.0a.{ordinal}, %{prefix}.linear.nooverflow.{ordinal}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.guard.2.{ordinal} = and i1 %{prefix}.guard.1.{ordinal}, %{prefix}.index.nooverflow.{ordinal}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.guard.{ordinal} = and i1 %{prefix}.guard.2.{ordinal}, %{prefix}.index.inbounds.{ordinal}"
        )
        .unwrap();
        let true_label = gfx950_stage_load_true_label(prefix, ordinal);
        let false_label = gfx950_stage_load_false_label(prefix, ordinal);
        let merge_label = gfx950_stage_load_merge_label(prefix, ordinal);
        writeln!(
            output,
            "  br i1 %{prefix}.guard.{ordinal}, label %{true_label}, label %{false_label}"
        )
        .unwrap();
        writeln!(output, "{true_label}:").unwrap();
        writeln!(
            output,
            "  %{prefix}.pointer.{ordinal} = getelementptr i8, ptr addrspace(1) {source_data}, i64 %{prefix}.index.{ordinal}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.loaded.true.{ordinal} = load i8, ptr addrspace(1) %{prefix}.pointer.{ordinal}, align 1"
        )
        .unwrap();
        writeln!(output, "  br label %{merge_label}").unwrap();
        writeln!(output, "{false_label}:").unwrap();
        writeln!(output, "  br label %{merge_label}").unwrap();
        writeln!(output, "{merge_label}:").unwrap();
        writeln!(
            output,
            "  %{prefix}.loaded.{ordinal} = phi i8 [ %{prefix}.loaded.true.{ordinal}, %{true_label} ], [ 0, %{false_label} ]"
        )
        .unwrap();
    }

    fn emit_gfx950_stage_store(
        &self,
        output: &mut dyn fmt::Write,
        prefix: &str,
        storage: &str,
        part: u32,
        byte: u32,
        value: &str,
    ) {
        let byte_offset = part * 8 + byte;
        writeln!(
            output,
            "  %{prefix}.lds.offset.{part}.{byte} = add i32 %{prefix}.lds.lane.base, {byte_offset}"
        )
        .unwrap();
        writeln!(
            output,
            "  %{prefix}.lds.pointer.{part}.{byte} = getelementptr i8, ptr addrspace(3) {storage}, i32 %{prefix}.lds.offset.{part}.{byte}"
        )
        .unwrap();
        writeln!(
            output,
            "  store i8 {value}, ptr addrspace(3) %{prefix}.lds.pointer.{part}.{byte}, align 1"
        )
        .unwrap();
    }

    fn emit_gfx950_lds_transpose_read(
        &self,
        output: &mut dyn fmt::Write,
        operation: &Operation,
        format: Gfx950LdsTransposeFormatV1,
        storage: ValueId,
    ) {
        let prefix = format!("v{}.transpose.read", operation.results[0].id.0);
        let storage = self.value(storage).0;
        self.emit_lane_id(output, &format!("%{prefix}.lane"), WaveWidth::Wave64);
        writeln!(
            output,
            "  %{prefix}.lane.base = mul i32 %{prefix}.lane, {}",
            format.lane_byte_stride()
        )
        .unwrap();
        let intrinsic = match format {
            Gfx950LdsTransposeFormatV1::Fp4E2M1 => AmdgcnIntrinsic::DsReadTr4B64,
            Gfx950LdsTransposeFormatV1::Fp8E4M3 => AmdgcnIntrinsic::DsReadTr8B64,
        };
        for part in 0..format.transpose_read_parts() {
            writeln!(
                output,
                "  %{prefix}.offset.{part} = add i32 %{prefix}.lane.base, {}",
                part * 8
            )
            .unwrap();
            writeln!(
                output,
                "  %{prefix}.pointer.{part} = getelementptr i8, ptr addrspace(3) {storage}, i32 %{prefix}.offset.{part}"
            )
            .unwrap();
            writeln!(
                output,
                "  %{prefix}.part.{part} = call <2 x i32> @{}(ptr addrspace(3) %{prefix}.pointer.{part})",
                intrinsic.llvm_name()
            )
            .unwrap();
            for element in 0..2u32 {
                let result_index = (part * 2 + element) as usize;
                writeln!(
                    output,
                    "  {} = extractelement <2 x i32> %{prefix}.part.{part}, i64 {element}",
                    value_name(operation.results[result_index].id)
                )
                .unwrap();
            }
        }
        for result in operation
            .results
            .iter()
            .skip((format.transpose_read_parts() * 2) as usize)
        {
            writeln!(output, "  {} = add i32 0, 0", value_name(result.id)).unwrap();
        }
    }

    fn emit_wave(&self, output: &mut dyn fmt::Write, result: &str, wave: &WaveOperation) {
        match wave.kind {
            WaveOperationKind::LaneId => self.emit_lane_id(output, result, wave.width),
            WaveOperationKind::Ballot { predicate } => {
                let predicate = self.value(predicate).0;
                let (ty, intrinsic) = ballot_intrinsic(wave.width);
                writeln!(
                    output,
                    "  {result} = call {ty} @{intrinsic}(i1 {predicate})"
                )
                .unwrap();
            }
            WaveOperationKind::Any { predicate } | WaveOperationKind::All { predicate } => {
                let predicate = self.value(predicate).0;
                let (ty, intrinsic) = ballot_intrinsic(wave.width);
                writeln!(
                    output,
                    "  {result}.mask = call {ty} @{intrinsic}(i1 {predicate})"
                )
                .unwrap();
                let comparison = if matches!(wave.kind, WaveOperationKind::Any { .. }) {
                    "ne"
                } else {
                    "eq"
                };
                let expected = if comparison == "ne" { "0" } else { "-1" };
                writeln!(
                    output,
                    "  {result} = icmp {comparison} {ty} {result}.mask, {expected}"
                )
                .unwrap();
            }
            WaveOperationKind::ShuffleIndex {
                value,
                source_lane,
                tile_width,
            } => {
                let value = self.value(value).0;
                let source_lane = self.value(source_lane).0;
                let lane = format!("{result}.lane");
                self.emit_lane_id(output, &lane, wave.width);
                writeln!(
                    output,
                    "  {result}.tile.base = and i32 {lane}, -{tile_width}"
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.tile.relative = and i32 {source_lane}, {}",
                    tile_width - 1
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.source = or i32 {result}.tile.base, {result}.tile.relative"
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.source.byte = shl i32 {result}.source, 2"
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result} = call i32 @{}(i32 {result}.source.byte, i32 {value})",
                    AmdgcnIntrinsic::DsBpermute.llvm_name()
                )
                .unwrap();
            }
            WaveOperationKind::ReduceF32 {
                value,
                tile_width,
                kind,
            } => {
                debug_assert!(tile_width != 0 && tile_width.is_power_of_two() && tile_width <= 64);
                let value = self.value(value).0;
                if tile_width == 1 {
                    writeln!(
                        output,
                        "  {result} = select i1 true, float {value}, float {value}"
                    )
                    .unwrap();
                    return;
                }
                let mut reduced = value.to_owned();
                self.emit_lane_id(output, &format!("{result}.lane"), wave.width);
                let stages = tile_width.trailing_zeros();
                for index in 0..stages {
                    let distance = 1_u32 << index;
                    writeln!(
                        output,
                        "  {result}.source.{index} = xor i32 {result}.lane, {distance}"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.source.byte.{index} = shl i32 {result}.source.{index}, 2"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.value.bits.{index} = bitcast float {reduced} to i32"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.remote.bits.{index} = call i32 @{}(i32 {result}.source.byte.{index}, i32 {result}.value.bits.{index})",
                        AmdgcnIntrinsic::DsBpermute.llvm_name()
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.remote.{index} = bitcast i32 {result}.remote.bits.{index} to float"
                    )
                    .unwrap();
                    let destination = if index + 1 == stages {
                        result.to_owned()
                    } else {
                        format!("{result}.reduce.{index}")
                    };
                    match kind {
                        WaveF32ReductionKindV1::Sum => writeln!(
                            output,
                            "  {destination} = fadd float {reduced}, {result}.remote.{index}"
                        )
                        .unwrap(),
                        WaveF32ReductionKindV1::Maximum => {
                            writeln!(
                                output,
                                "  {result}.less.{index} = fcmp olt float {reduced}, {result}.remote.{index}"
                            )
                            .unwrap();
                            writeln!(
                                output,
                                "  {destination} = select i1 {result}.less.{index}, float {result}.remote.{index}, float {reduced}"
                            )
                            .unwrap();
                        }
                    }
                    reduced = destination;
                }
            }
            WaveOperationKind::BroadcastF32 {
                value,
                source_lane,
                tile_width,
            } => {
                debug_assert!(tile_width != 0 && tile_width.is_power_of_two() && tile_width <= 64);
                let value = self.value(value).0;
                let source_lane = self.value(source_lane).0;
                self.emit_lane_id(output, &format!("{result}.lane"), wave.width);
                writeln!(
                    output,
                    "  {result}.tile.base = and i32 {result}.lane, {}",
                    -(tile_width as i32),
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.source = add i32 {result}.tile.base, {source_lane}"
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.source.byte = shl i32 {result}.source, 2"
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.value.bits = bitcast float {value} to i32"
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result}.bits = call i32 @{}(i32 {result}.source.byte, i32 {result}.value.bits)",
                    AmdgcnIntrinsic::DsBpermute.llvm_name()
                )
                .unwrap();
                writeln!(output, "  {result} = bitcast i32 {result}.bits to float").unwrap();
            }
        }
    }

    fn emit_diagnostic(
        &self,
        output: &mut dyn fmt::Write,
        result: Option<&str>,
        diagnostic: &AmdGpuDiagnosticOperation,
    ) {
        match diagnostic {
            AmdGpuDiagnosticOperation::Clock32 => {
                let result = result.expect("validated clock result");
                writeln!(
                    output,
                    "  {result}.i64 = call i64 @llvm.amdgcn.s.memrealtime()"
                )
                .unwrap();
                writeln!(output, "  {result} = trunc i64 {result}.i64 to i32").unwrap();
            }
            AmdGpuDiagnosticOperation::Trap => {
                writeln!(output, "  call void @llvm.trap()").unwrap();
            }
            AmdGpuDiagnosticOperation::DebugTrap => {
                writeln!(output, "  call void @llvm.debugtrap()").unwrap();
            }
            AmdGpuDiagnosticOperation::ProfilingMarker { marker } => {
                Self::emit_marker_word(
                    output,
                    self.constant_u32(*marker).expect("validated marker"),
                    false,
                );
            }
            AmdGpuDiagnosticOperation::Print {
                format_id,
                arguments,
            } => {
                Self::emit_marker_word(
                    output,
                    self.constant_u32(*format_id)
                        .expect("validated format identity"),
                    true,
                );
                for argument in arguments {
                    let value = self.value(*argument).0;
                    writeln!(
                        output,
                        "  call void asm sideeffect \"s_nop 0\", \"v\"(i32 {value})"
                    )
                    .unwrap();
                }
            }
            AmdGpuDiagnosticOperation::AssertFail { site_id, line } => {
                Self::emit_marker_word(
                    output,
                    self.constant_u32(*site_id)
                        .expect("validated assertion site"),
                    true,
                );
                Self::emit_marker_word(
                    output,
                    self.constant_u32(*line).expect("validated assertion line"),
                    true,
                );
                writeln!(output, "  call void @llvm.trap()").unwrap();
            }
        }
    }

    fn emit_marker_word(output: &mut dyn fmt::Write, value: u32, high_half: bool) {
        writeln!(
            output,
            "  call void asm sideeffect \"s_nop {}\", \"\"()",
            value & u32::from(u16::MAX)
        )
        .unwrap();
        if high_half {
            writeln!(
                output,
                "  call void asm sideeffect \"s_nop {}\", \"\"()",
                value >> 16
            )
            .unwrap();
        }
    }

    fn emit_float(&self, output: &mut dyn fmt::Write, result: &str, float: &FloatOperation) {
        match float {
            FloatOperation::Convert { kind, value } => {
                let value = self.value(*value).0;
                let (result_ty, helper, operand_ty) = match kind {
                    FloatConversionKind::F16ToF32 => ("float", "__fe2o3_f16_to_f32_v1", "i16"),
                    FloatConversionKind::F32ToF16RoundTiesEven => {
                        ("i16", "__fe2o3_f32_to_f16_rne_v1", "float")
                    }
                    FloatConversionKind::Bf16ToF32 => ("float", "__fe2o3_bf16_to_f32_v1", "i16"),
                    FloatConversionKind::F32ToBf16RoundTiesEven => {
                        ("i16", "__fe2o3_f32_to_bf16_rne_v1", "float")
                    }
                };
                writeln!(
                    output,
                    "  {result} = call {result_ty} @{helper}({operand_ty} {value})"
                )
                .unwrap();
            }
            FloatOperation::WidenedBinary {
                format,
                op,
                lhs,
                rhs,
            } => {
                let lhs = self.value(*lhs).0;
                let rhs = self.value(*rhs).0;
                let (widen, narrow) = narrow_float_helpers(*format);
                writeln!(output, "  {result}.lhs = call float @{widen}(i16 {lhs})").unwrap();
                writeln!(output, "  {result}.rhs = call float @{widen}(i16 {rhs})").unwrap();
                writeln!(
                    output,
                    "  {result}.f32 = call float @{}(float {result}.lhs, float {result}.rhs, metadata !\"round.tonearest\", metadata !\"fpexcept.ignore\")",
                    constrained_binary_name(*op)
                )
                .unwrap();
                writeln!(
                    output,
                    "  {result} = call i16 @{narrow}(float {result}.f32)"
                )
                .unwrap();
            }
            FloatOperation::F32Math {
                function,
                implementation,
                arguments,
            } => match implementation {
                F32MathImplementation::IeeeSqrtRoundTiesEvenIgnoreExceptionsV1 => {
                    let [argument] = arguments.as_slice() else {
                        unreachable!("verifier checked sqrt arity")
                    };
                    let metadata = if self.target == LoweringTarget::Gfx950XnackMinusV1 {
                        ""
                    } else {
                        ", metadata !\"round.tonearest\", metadata !\"fpexcept.ignore\""
                    };
                    writeln!(
                        output,
                        "  {result} = call float @{}(float {}{metadata})",
                        constrained_math_name(*function, self.target),
                        self.value(*argument).0
                    )
                    .unwrap();
                }
                F32MathImplementation::ConstrainedLlvm => {
                    let arguments = arguments
                        .iter()
                        .map(|argument| format!("float {}", self.value(*argument).0))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let metadata = match function {
                        F32MathFunction::FusedMultiplyAdd => {
                            ", metadata !\"round.tonearest\", metadata !\"fpexcept.ignore\""
                        }
                        F32MathFunction::Floor
                        | F32MathFunction::Ceil
                        | F32MathFunction::Truncate
                        | F32MathFunction::RoundTiesEven => ", metadata !\"fpexcept.ignore\"",
                        _ => unreachable!("verifier fixed the math implementation"),
                    };
                    writeln!(
                        output,
                        "  {result} = call float @{}({arguments}{metadata})",
                        constrained_math_name(*function, self.target)
                    )
                    .unwrap();
                }
                F32MathImplementation::OcmlAbiV1 => {
                    let [argument] = arguments.as_slice() else {
                        unreachable!("verifier checked OCML arity")
                    };
                    writeln!(
                        output,
                        "  {result} = call float @{}(float {})",
                        ocml_name(*function),
                        self.value(*argument).0
                    )
                    .unwrap();
                }
            },
            FloatOperation::Bf16x2FusedMultiplyAdd {
                value,
                multiplier,
                addend,
            } => {
                let packed =
                    [*value, *multiplier, *addend].map(|value| self.value(value).0.to_string());
                for (name, value) in ["value", "multiplier", "addend"].into_iter().zip(packed) {
                    writeln!(output, "  {result}.{name}.lo = trunc i32 {value} to i16").unwrap();
                    writeln!(output, "  {result}.{name}.shift = lshr i32 {value}, 16").unwrap();
                    writeln!(
                        output,
                        "  {result}.{name}.hi = trunc i32 {result}.{name}.shift to i16"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.{name}.0 = call float @__fe2o3_bf16_to_f32_v1(i16 {result}.{name}.lo)"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.{name}.1 = call float @__fe2o3_bf16_to_f32_v1(i16 {result}.{name}.hi)"
                    )
                    .unwrap();
                }
                for lane in 0..2 {
                    writeln!(
                        output,
                        "  {result}.fma.{lane} = call float @llvm.experimental.constrained.fma.f32(float {result}.value.{lane}, float {result}.multiplier.{lane}, float {result}.addend.{lane}, metadata !\"round.tonearest\", metadata !\"fpexcept.ignore\")"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.bf16.{lane} = call i16 @__fe2o3_f32_to_bf16_rne_v1(float {result}.fma.{lane})"
                    )
                    .unwrap();
                    writeln!(
                        output,
                        "  {result}.wide.{lane} = zext i16 {result}.bf16.{lane} to i32"
                    )
                    .unwrap();
                }
                writeln!(output, "  {result}.high = shl i32 {result}.wide.1, 16").unwrap();
                writeln!(output, "  {result} = or i32 {result}.wide.0, {result}.high").unwrap();
            }
        }
    }

    fn emit_lane_id(&self, output: &mut dyn fmt::Write, result: &str, width: WaveWidth) {
        writeln!(
            output,
            "  {result}.lo = call i32 @{}(i32 -1, i32 0)",
            AmdgcnIntrinsic::MbcntLo.llvm_name()
        )
        .unwrap();
        match width {
            WaveWidth::Wave32 => {
                writeln!(output, "  {result} = add i32 {result}.lo, 0").unwrap();
            }
            WaveWidth::Wave64 => {
                writeln!(
                    output,
                    "  {result} = call i32 @{}(i32 -1, i32 {result}.lo)",
                    AmdgcnIntrinsic::MbcntHi.llvm_name()
                )
                .unwrap();
            }
        }
    }

    fn validate_atomic(
        &self,
        atomic: &Atomic,
        location: &LoweringLocation,
    ) -> Result<(), LoweringErrors> {
        let pointer = self.value_type(atomic.pointer);
        validate_memory_access(pointer, atomic.access.address_space, location, self.target)?;
        let Type::Pointer(pointer) = pointer else {
            unreachable!("kernel IR verification required an atomic pointer")
        };
        let Some(scalar) = pointer.pointee.as_scalar() else {
            unreachable!("kernel IR verification required a scalar atomic pointee")
        };

        if !matches!(
            scalar,
            ScalarType::I32 | ScalarType::U32 | ScalarType::I64 | ScalarType::U64
        ) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedAtomic,
                format!(
                    "AMDGPU atomic lowering supports only 32-bit and 64-bit integers, found {scalar:?}"
                ),
            ));
        }
        if atomic.access.volatile {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedAtomic,
                "volatile scoped atomics are outside the supported AMDGPU subset",
            ));
        }
        if !supported_atomic_address_scope(atomic.access.address_space, atomic.scope) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedAtomic,
                format!(
                    "AMDGPU atomic lowering does not support {:?} memory at {:?} scope",
                    atomic.access.address_space, atomic.scope
                ),
            ));
        }
        Ok(())
    }

    fn emit_atomic(&self, output: &mut dyn fmt::Write, operation: &Operation, atomic: &Atomic) {
        let (pointer_name, pointer_ty) = self.value(atomic.pointer);
        let Type::Pointer(pointer_ty) = pointer_ty else {
            unreachable!("atomic preflight required a pointer")
        };
        let value_type = llvm_type(&pointer_ty.pointee);
        let address_space = llvm_address_space(pointer_ty.address_space);
        let sync_scope = llvm_atomic_sync_scope(atomic.scope);
        let ordering = llvm_atomic_ordering(atomic.ordering);

        match atomic.kind {
            AtomicKind::Load => {
                let result = operation
                    .results
                    .first()
                    .expect("verified atomic load result");
                writeln!(
                    output,
                    "  {} = load atomic {value_type}, ptr addrspace({address_space}) {pointer_name}{sync_scope} {ordering}, align {}",
                    value_name(result.id),
                    atomic.access.alignment
                )
                .unwrap();
            }
            AtomicKind::Store => {
                let value = self
                    .value(atomic.value.expect("verified atomic store value"))
                    .0;
                writeln!(
                    output,
                    "  store atomic {value_type} {value}, ptr addrspace({address_space}) {pointer_name}{sync_scope} {ordering}, align {}",
                    atomic.access.alignment
                )
                .unwrap();
            }
            AtomicKind::CompareExchange => {
                let [old, succeeded] = operation.results.as_slice() else {
                    unreachable!("verified compare-exchange results")
                };
                let desired = self
                    .value(
                        atomic
                            .value
                            .expect("verified compare-exchange desired value"),
                    )
                    .0;
                let expected = self
                    .value(
                        atomic
                            .compare
                            .expect("verified compare-exchange expected value"),
                    )
                    .0;
                let failure_ordering = llvm_atomic_ordering(
                    atomic
                        .failure_ordering
                        .expect("verified compare-exchange failure ordering"),
                );
                let pair = format!("{}.cmpxchg", value_name(old.id));
                writeln!(
                    output,
                    "  {pair} = cmpxchg ptr addrspace({address_space}) {pointer_name}, {value_type} {expected}, {value_type} {desired}{sync_scope} {ordering} {failure_ordering}, align {}",
                    atomic.access.alignment
                )
                .unwrap();
                writeln!(
                    output,
                    "  {} = extractvalue {{ {value_type}, i1 }} {pair}, 0",
                    value_name(old.id)
                )
                .unwrap();
                writeln!(
                    output,
                    "  {} = extractvalue {{ {value_type}, i1 }} {pair}, 1",
                    value_name(succeeded.id)
                )
                .unwrap();
            }
            kind => {
                let result = operation
                    .results
                    .first()
                    .expect("verified atomic RMW result");
                let value = self
                    .value(atomic.value.expect("verified atomic RMW value"))
                    .0;
                let scalar = pointer_ty
                    .pointee
                    .as_scalar()
                    .expect("atomic preflight required a scalar");
                let opcode = llvm_atomic_rmw_opcode(kind, scalar);
                writeln!(
                    output,
                    "  {} = atomicrmw {opcode} ptr addrspace({address_space}) {pointer_name}, {value_type} {value}{sync_scope} {ordering}, align {}",
                    value_name(result.id),
                    atomic.access.alignment
                )
                .unwrap();
            }
        }
    }

    fn emit_terminator(
        &self,
        output: &mut dyn fmt::Write,
        predecessor: BlockId,
        terminator: &Terminator,
    ) {
        match terminator {
            Terminator::Branch { target, .. } => {
                writeln!(
                    output,
                    "  br label %{}",
                    self.edge_target_label(predecessor, 0, *target)
                )
                .unwrap();
            }
            Terminator::ConditionalBranch {
                condition,
                then_target,
                else_target,
                ..
            } => {
                let (condition, _) = self.value(*condition);
                writeln!(
                    output,
                    "  br i1 {condition}, label %{}, label %{}",
                    self.edge_target_label(predecessor, 0, *then_target),
                    self.edge_target_label(predecessor, 1, *else_target)
                )
                .unwrap();
            }
            Terminator::Switch {
                selector,
                cases,
                default_target,
                ..
            } => {
                let (selector, ty) = self.value(*selector);
                writeln!(
                    output,
                    "  switch {} {selector}, label %{} [",
                    llvm_type(ty),
                    self.edge_target_label(predecessor, cases.len(), *default_target)
                )
                .unwrap();
                for (ordinal, case) in cases.iter().enumerate() {
                    writeln!(
                        output,
                        "    {} {}, label %{}",
                        llvm_type(ty),
                        case.value,
                        self.edge_target_label(predecessor, ordinal, case.target)
                    )
                    .unwrap();
                }
                writeln!(output, "  ]").unwrap();
            }
            Terminator::IntegerSwitch {
                selector,
                cases,
                default_target,
                ..
            } => {
                let (selector, ty) = self.value(*selector);
                writeln!(
                    output,
                    "  switch {} {selector}, label %{} [",
                    llvm_type(ty),
                    self.edge_target_label(predecessor, cases.len(), *default_target)
                )
                .unwrap();
                for (ordinal, case) in cases.iter().enumerate() {
                    writeln!(
                        output,
                        "    {} {}, label %{}",
                        llvm_type(ty),
                        constant_value(&case.value)
                            .expect("verified integer switch case has an integer constant"),
                        self.edge_target_label(predecessor, ordinal, case.target)
                    )
                    .unwrap();
                }
                writeln!(output, "  ]").unwrap();
            }
            Terminator::Return { values } => match values.as_slice() {
                [] => writeln!(output, "  ret void").unwrap(),
                [value] => {
                    let (name, ty) = self.value(*value);
                    writeln!(output, "  ret {} {name}", llvm_type(ty)).unwrap();
                }
                _ => unreachable!("compiler-module preflight rejected multi-value returns"),
            },
            Terminator::Unreachable => writeln!(output, "  unreachable").unwrap(),
        }
    }

    fn emit_split_edges(&self, output: &mut dyn fmt::Write, block: &BasicBlock) {
        let outgoing = self
            .control_flow
            .outgoing_edges(block.id)
            .expect("indexed source block");
        for edge_index in outgoing {
            if !self.split_edges[edge_index] {
                continue;
            }
            let edge = self.control_flow.edge(edge_index).expect("indexed edge");
            let target = self
                .control_flow
                .edge_target(edge_index)
                .expect("indexed edge target");
            let ordinal = edge.ordinal();
            writeln!(output, "{}:", edge_label(block.id, ordinal, target)).unwrap();
            writeln!(output, "  br label %{}", block_label(target)).unwrap();
        }
    }
}

fn supported_scalar(scalar: ScalarType, target: LoweringTarget) -> bool {
    scalar == ScalarType::Bool
        || scalar.is_integer()
        || scalar == ScalarType::F32
        || (target.supports_narrow_float() && matches!(scalar, ScalarType::F16 | ScalarType::Bf16))
}

fn is_i32_register_type(ty: &Type) -> bool {
    matches!(ty, Type::Scalar(ScalarType::I32 | ScalarType::U32))
}

fn supported_integer(scalar: ScalarType) -> bool {
    matches!(
        scalar,
        ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::U8
            | ScalarType::U16
            | ScalarType::U32
            | ScalarType::U64
            | ScalarType::Index
    )
}

fn supported_memory_type(ty: &Type, target: LoweringTarget) -> bool {
    matches!(ty, Type::Scalar(scalar) if supported_integer(*scalar)
        || *scalar == ScalarType::F32
        || (target.supports_narrow_float()
            && matches!(scalar, ScalarType::F16 | ScalarType::Bf16)))
}

fn supported_private_memory_type(ty: &Type, target: LoweringTarget) -> bool {
    supported_memory_type(ty, target)
        || matches!(
            ty,
            Type::Pointer(pointer)
                if matches!(
                    pointer.address_space,
                    KernelAddressSpace::Global | KernelAddressSpace::Workgroup
                ) && (pointer.pointee.as_ref() == &Type::Unit
                    || supported_memory_type(&pointer.pointee, target))
        )
}

fn amdgpu_private_element_alignment(ty: &Type) -> Option<u64> {
    match ty {
        Type::Scalar(ScalarType::Bool | ScalarType::I8 | ScalarType::U8) => Some(1),
        Type::Scalar(ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::Bf16) => {
            Some(2)
        }
        Type::Scalar(ScalarType::I32 | ScalarType::U32 | ScalarType::F32) => Some(4),
        Type::Scalar(ScalarType::I64 | ScalarType::U64 | ScalarType::Index) | Type::Pointer(_) => {
            Some(8)
        }
        _ => None,
    }
}

fn amdgpu_lds_element_bytes(ty: &Type) -> Option<u64> {
    match ty {
        Type::Scalar(ScalarType::I8 | ScalarType::U8) => Some(1),
        Type::Scalar(ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::Bf16) => {
            Some(2)
        }
        Type::Scalar(ScalarType::I32 | ScalarType::U32 | ScalarType::F32) => Some(4),
        Type::Scalar(ScalarType::I64 | ScalarType::U64 | ScalarType::Index) => Some(8),
        _ => None,
    }
}

fn supported_atomic_capability(
    width_bits: u16,
    address_space: KernelAddressSpace,
    max_scope: SynchronizationScope,
) -> bool {
    matches!(width_bits, 32 | 64) && supported_atomic_address_scope(address_space, max_scope)
}

fn supported_atomic_address_scope(
    address_space: KernelAddressSpace,
    scope: SynchronizationScope,
) -> bool {
    match address_space {
        KernelAddressSpace::Workgroup => scope == SynchronizationScope::Workgroup,
        KernelAddressSpace::Global => matches!(
            scope,
            SynchronizationScope::Workgroup
                | SynchronizationScope::Device
                | SynchronizationScope::System
        ),
        KernelAddressSpace::Generic
        | KernelAddressSpace::Private
        | KernelAddressSpace::Constant => false,
    }
}

fn supported_binary(op: BinaryOp, ty: &Type, target: LoweringTarget) -> bool {
    let Some(scalar) = ty.as_scalar() else {
        return false;
    };
    match op {
        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => {
            supported_integer(scalar) || scalar == ScalarType::F32
        }
        BinaryOp::Divide if scalar == ScalarType::F32 => {
            !matches!(target, LoweringTarget::Baseline)
        }
        BinaryOp::Divide | BinaryOp::Remainder => supported_integer(scalar),
        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
            scalar == ScalarType::Bool || supported_integer(scalar)
        }
        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => supported_integer(scalar),
        BinaryOp::Checked(_) => scalar.is_integer(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LlvmUnaryStyle {
    FloatNegate,
    SignedNegate,
    BooleanNot,
    IntegerNot,
}

fn unary_lowering_style(op: UnaryOp, ty: &Type) -> Option<LlvmUnaryStyle> {
    let scalar = ty.as_scalar()?;
    match (op, scalar) {
        (UnaryOp::Negate, ScalarType::F32) => Some(LlvmUnaryStyle::FloatNegate),
        (UnaryOp::Negate, scalar) if scalar.is_signed_integer() && supported_integer(scalar) => {
            Some(LlvmUnaryStyle::SignedNegate)
        }
        (UnaryOp::Not, ScalarType::Bool) => Some(LlvmUnaryStyle::BooleanNot),
        (UnaryOp::Not, scalar) if supported_integer(scalar) => Some(LlvmUnaryStyle::IntegerNot),
        _ => None,
    }
}

fn supported_unary(op: UnaryOp, ty: &Type) -> bool {
    unary_lowering_style(op, ty).is_some()
}

fn validate_pointer(
    ty: &Type,
    location: &LoweringLocation,
    target: LoweringTarget,
) -> Result<(), LoweringErrors> {
    let Type::Pointer(pointer) = ty else {
        unreachable!("verify_module checked GEP base")
    };
    if !matches!(
        pointer.address_space,
        KernelAddressSpace::Private | KernelAddressSpace::Global | KernelAddressSpace::Workgroup
    ) {
        return Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedAddressSpace,
            format!(
                "G1 supports only private, global, or workgroup pointers, found {:?}",
                pointer.address_space
            ),
        ));
    }
    let supported = if pointer.address_space == KernelAddressSpace::Private {
        supported_private_memory_type(&pointer.pointee, target)
    } else {
        supported_memory_type(&pointer.pointee, target)
    };
    if !supported {
        return Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedType,
            format!("unsupported memory pointee type {:?}", pointer.pointee),
        ));
    }
    Ok(())
}

fn validate_device_pointer(
    ty: &Type,
    location: &LoweringLocation,
    target: LoweringTarget,
) -> Result<(), LoweringErrors> {
    let Type::Pointer(pointer) = ty else {
        unreachable!("device pointer validation requires a pointer")
    };
    if !matches!(
        pointer.address_space,
        KernelAddressSpace::Global | KernelAddressSpace::Workgroup
    ) {
        return Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedAddressSpace,
            format!(
                "G1 supports only global or workgroup pointers, found {:?}",
                pointer.address_space
            ),
        ));
    }
    if pointer.pointee.as_ref() != &Type::Unit && !supported_memory_type(&pointer.pointee, target) {
        return Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedType,
            format!("unsupported memory pointee type {:?}", pointer.pointee),
        ));
    }
    Ok(())
}

fn validate_memory_intrinsic(
    intrinsic: &MemoryIntrinsicOperation,
    location: &LoweringLocation,
    target: LoweringTarget,
) -> Result<(), LoweringErrors> {
    let (element, address_spaces) = match intrinsic {
        MemoryIntrinsicOperation::PointerDistance {
            element,
            address_space,
            ..
        }
        | MemoryIntrinsicOperation::VolatileLoad {
            element,
            address_space,
            ..
        }
        | MemoryIntrinsicOperation::VolatileStore {
            element,
            address_space,
            ..
        } => (*element, [Some(*address_space), None]),
        MemoryIntrinsicOperation::CopyNonOverlapping {
            element,
            source_address_space,
            destination_address_space,
            ..
        } => (
            *element,
            [
                Some(*source_address_space),
                Some(*destination_address_space),
            ],
        ),
    };
    for address_space in address_spaces.into_iter().flatten() {
        if !matches!(
            address_space,
            KernelAddressSpace::Global | KernelAddressSpace::Workgroup
        ) {
            return Err(LoweringErrors::one(
                location.clone(),
                LoweringDiagnosticCode::UnsupportedAddressSpace,
                format!(
                    "gfx942 memory intrinsics support only global or workgroup pointers, found {address_space:?}"
                ),
            ));
        }
    }
    if element != MemoryElementType::Unit && !supported_memory_type(&element.ir_type(), target) {
        return Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedType,
            format!(
                "gfx942 memory intrinsic does not support element type {:?}",
                element.ir_type()
            ),
        ));
    }
    Ok(())
}

fn validate_memory_access(
    pointer: &Type,
    access_space: KernelAddressSpace,
    location: &LoweringLocation,
    target: LoweringTarget,
) -> Result<(), LoweringErrors> {
    validate_pointer(pointer, location, target)?;
    let Type::Pointer(pointer) = pointer else {
        unreachable!("validate_pointer required a pointer")
    };
    if access_space != pointer.address_space {
        return Err(LoweringErrors::one(
            location.clone(),
            LoweringDiagnosticCode::UnsupportedAddressSpace,
            format!(
                "memory access names {access_space:?} but pointer uses {:?}",
                pointer.address_space
            ),
        ));
    }
    Ok(())
}

fn validate_constant(constant: &Constant, target: LoweringTarget) -> Result<(), String> {
    match constant {
        Constant::Bool(_)
        | Constant::I8(_)
        | Constant::I16(_)
        | Constant::I32(_)
        | Constant::I64(_)
        | Constant::U8(_)
        | Constant::U16(_)
        | Constant::U32(_)
        | Constant::U64(_)
        | Constant::Index(_) => Ok(()),
        Constant::F32Bits(bits) if !f32::from_bits(*bits).is_nan() => Ok(()),
        Constant::F32Bits(_) => Err("G1 rejects NaN f32 constants because LLVM's widened hexadecimal spelling does not preserve every payload".to_string()),
        Constant::F16Bits(_) | Constant::Bf16Bits(_) if target.supports_narrow_float() => Ok(()),
        _ => Err(format!("G1 does not lower constant {constant:?}")),
    }
}

fn validate_cast(
    kind: CastKind,
    from: &Type,
    to: &Type,
    target: LoweringTarget,
) -> Result<(), String> {
    let (Some(from_scalar), Some(to_scalar)) = (from.as_scalar(), to.as_scalar()) else {
        return Err(format!(
            "G1 casts require scalar types, found {from:?} to {to:?}"
        ));
    };
    if !supported_scalar(from_scalar, target) || !supported_scalar(to_scalar, target) {
        return Err(format!("G1 does not lower cast types {from:?} to {to:?}"));
    }
    let from_width = llvm_width(from_scalar);
    let to_width = llvm_width(to_scalar);
    let valid = match kind {
        CastKind::Truncate => {
            supported_integer(from_scalar) && supported_integer(to_scalar) && from_width > to_width
        }
        CastKind::ZeroExtend => {
            (from_scalar == ScalarType::Bool
                || (!from_scalar.is_signed_integer() && supported_integer(from_scalar)))
                && supported_integer(to_scalar)
                && from_width < to_width
        }
        CastKind::SignExtend => {
            from_scalar.is_signed_integer() && supported_integer(to_scalar) && from_width < to_width
        }
        CastKind::IntegerToFloat => supported_integer(from_scalar) && to_scalar == ScalarType::F32,
        CastKind::FloatToInteger => from_scalar == ScalarType::F32 && supported_integer(to_scalar),
        CastKind::Bitcast => from_width == to_width && from_scalar != to_scalar,
        CastKind::FloatExtend | CastKind::FloatTruncate => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("unsupported {kind:?} cast from {from:?} to {to:?}"))
    }
}

fn value_name(value: ValueId) -> String {
    format!("%v{}", value.0)
}

fn lds_symbol(kernel: &Kernel, value: ValueId) -> String {
    format!("@__fe2o3_lds_{}_{}", kernel.id.as_str(), value.0)
}

fn emit_fence(
    output: &mut dyn fmt::Write,
    scope: fe2o3_kernel_ir::SynchronizationScope,
    ordering: MemoryOrdering,
) {
    let ordering = match ordering {
        MemoryOrdering::Acquire => "acquire",
        MemoryOrdering::Release => "release",
        MemoryOrdering::AcquireRelease => "acq_rel",
        MemoryOrdering::SequentiallyConsistent => "seq_cst",
        MemoryOrdering::Relaxed => unreachable!("verification rejected a relaxed fence"),
    };
    match scope {
        fe2o3_kernel_ir::SynchronizationScope::Subgroup => {
            writeln!(output, "  fence syncscope(\"wavefront\") {ordering}").unwrap();
        }
        fe2o3_kernel_ir::SynchronizationScope::Workgroup => {
            writeln!(output, "  fence syncscope(\"workgroup\") {ordering}").unwrap();
        }
        fe2o3_kernel_ir::SynchronizationScope::Device => {
            writeln!(output, "  fence syncscope(\"agent\") {ordering}").unwrap();
        }
        fe2o3_kernel_ir::SynchronizationScope::System => {
            writeln!(output, "  fence {ordering}").unwrap();
        }
        fe2o3_kernel_ir::SynchronizationScope::Invocation => {
            unreachable!("verification rejected invocation-scoped synchronization")
        }
    }
}

fn llvm_atomic_sync_scope(scope: SynchronizationScope) -> &'static str {
    match scope {
        SynchronizationScope::Workgroup => " syncscope(\"workgroup\")",
        SynchronizationScope::Device => " syncscope(\"agent\")",
        SynchronizationScope::System => "",
        SynchronizationScope::Invocation | SynchronizationScope::Subgroup => {
            unreachable!("atomic preflight rejected unsupported synchronization scope")
        }
    }
}

fn llvm_atomic_ordering(ordering: MemoryOrdering) -> &'static str {
    match ordering {
        MemoryOrdering::Relaxed => "monotonic",
        MemoryOrdering::Acquire => "acquire",
        MemoryOrdering::Release => "release",
        MemoryOrdering::AcquireRelease => "acq_rel",
        MemoryOrdering::SequentiallyConsistent => "seq_cst",
    }
}

fn llvm_atomic_rmw_opcode(kind: AtomicKind, scalar: ScalarType) -> &'static str {
    match kind {
        AtomicKind::Exchange => "xchg",
        AtomicKind::Add => "add",
        AtomicKind::Subtract => "sub",
        AtomicKind::Min if scalar.is_signed_integer() => "min",
        AtomicKind::Min => "umin",
        AtomicKind::Max if scalar.is_signed_integer() => "max",
        AtomicKind::Max => "umax",
        AtomicKind::BitAnd => "and",
        AtomicKind::BitOr => "or",
        AtomicKind::BitXor => "xor",
        AtomicKind::Load | AtomicKind::Store | AtomicKind::CompareExchange => {
            unreachable!("non-RMW atomic kind")
        }
    }
}

fn ballot_intrinsic(width: WaveWidth) -> (&'static str, &'static str) {
    match width {
        WaveWidth::Wave32 => ("i32", AmdgcnIntrinsic::Ballot32.llvm_name()),
        WaveWidth::Wave64 => ("i64", AmdgcnIntrinsic::Ballot64.llvm_name()),
    }
}

fn block_label(block: BlockId) -> String {
    format!("bb{}", block.0)
}

fn guarded_load_true_label(block: BlockId, operation: usize) -> String {
    format!("guarded_load_bb{}_op{}_true", block.0, operation)
}

fn guarded_load_false_label(block: BlockId, operation: usize) -> String {
    format!("guarded_load_bb{}_op{}_false", block.0, operation)
}

fn guarded_load_merge_label(block: BlockId, operation: usize) -> String {
    format!("guarded_load_bb{}_op{}_merge", block.0, operation)
}

fn gfx950_stage_load_true_label(prefix: &str, ordinal: u32) -> String {
    format!("{prefix}.load.{ordinal}.true")
}

fn gfx950_stage_load_false_label(prefix: &str, ordinal: u32) -> String {
    format!("{prefix}.load.{ordinal}.false")
}

fn gfx950_stage_load_merge_label(prefix: &str, ordinal: u32) -> String {
    format!("{prefix}.load.{ordinal}.merge")
}

fn edge_label(predecessor: BlockId, ordinal: usize, target: BlockId) -> String {
    format!("edge_bb{}_{}_bb{}", predecessor.0, ordinal, target.0)
}

fn llvm_type(ty: &Type) -> &'static str {
    match ty {
        Type::Scalar(scalar) => llvm_scalar(*scalar),
        Type::Pointer(pointer) if pointer.address_space == KernelAddressSpace::Global => {
            "ptr addrspace(1)"
        }
        Type::Pointer(pointer) if pointer.address_space == KernelAddressSpace::Workgroup => {
            "ptr addrspace(3)"
        }
        Type::Pointer(pointer) if pointer.address_space == KernelAddressSpace::Private => {
            "ptr addrspace(5)"
        }
        Type::Pointer(_) => unreachable!("preflight rejected unsupported address space"),
        Type::Unit | Type::Slice(_) => unreachable!("type is not a first-class G1 LLVM value"),
    }
}

fn llvm_address_space(address_space: KernelAddressSpace) -> u32 {
    match address_space {
        KernelAddressSpace::Global => 1,
        KernelAddressSpace::Workgroup => 3,
        KernelAddressSpace::Private => 5,
        _ => unreachable!("preflight rejected unsupported address space"),
    }
}

fn llvm_scalar(scalar: ScalarType) -> &'static str {
    match scalar {
        ScalarType::Bool => "i1",
        ScalarType::I8 | ScalarType::U8 => "i8",
        ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::Bf16 => "i16",
        ScalarType::I32 | ScalarType::U32 => "i32",
        ScalarType::I64 | ScalarType::U64 | ScalarType::Index => "i64",
        ScalarType::F32 => "float",
        ScalarType::I128 | ScalarType::U128 => "i128",
        ScalarType::F64 => unreachable!("preflight rejected unsupported scalar"),
    }
}

fn llvm_width(scalar: ScalarType) -> u16 {
    scalar.bit_width().unwrap_or(64)
}

fn constant_value(constant: &Constant) -> Option<String> {
    match constant {
        Constant::Bool(value) => Some(value.to_string()),
        Constant::I8(value) => Some(value.to_string()),
        Constant::I16(value) => Some(value.to_string()),
        Constant::I32(value) => Some(value.to_string()),
        Constant::I64(value) => Some(value.to_string()),
        Constant::U8(value) => Some(value.to_string()),
        Constant::U16(value) => Some(value.to_string()),
        Constant::U32(value) => Some(value.to_string()),
        Constant::U64(value) => Some(value.to_string()),
        Constant::Index(value) => Some(value.to_string()),
        Constant::F16Bits(value) | Constant::Bf16Bits(value) => Some(value.to_string()),
        Constant::F32Bits(bits) if !f32::from_bits(*bits).is_nan() => Some(format!(
            "0x{:016X}",
            f64::from(f32::from_bits(*bits)).to_bits()
        )),
        _ => None,
    }
}

fn binary_opcode(op: BinaryOp, ty: &Type) -> &'static str {
    let floating = ty.as_scalar() == Some(ScalarType::F32);
    match (op, floating) {
        (BinaryOp::Add, false) => "add",
        (BinaryOp::Subtract, false) => "sub",
        (BinaryOp::Multiply, false) => "mul",
        (BinaryOp::Divide, false) if ty.as_scalar().is_some_and(ScalarType::is_signed_integer) => {
            "sdiv"
        }
        (BinaryOp::Divide, false) => "udiv",
        (BinaryOp::Remainder, false)
            if ty.as_scalar().is_some_and(ScalarType::is_signed_integer) =>
        {
            "srem"
        }
        (BinaryOp::Remainder, false) => "urem",
        (BinaryOp::BitAnd, false) => "and",
        (BinaryOp::BitOr, false) => "or",
        (BinaryOp::BitXor, false) => "xor",
        (BinaryOp::ShiftLeft, false) => "shl",
        (BinaryOp::ShiftRight, false)
            if ty.as_scalar().is_some_and(ScalarType::is_signed_integer) =>
        {
            "ashr"
        }
        (BinaryOp::ShiftRight, false) => "lshr",
        (BinaryOp::Add, true) => "fadd",
        (BinaryOp::Subtract, true) => "fsub",
        (BinaryOp::Multiply, true) => "fmul",
        (BinaryOp::Divide, true) => "fdiv",
        (BinaryOp::Checked(_), _) => {
            unreachable!("checked binary operations use LLVM overflow intrinsics")
        }
        _ => unreachable!("preflight rejected unsupported binary operation"),
    }
}

fn checked_binary_intrinsic_name(operator: CheckedBinaryOperator, scalar: ScalarType) -> String {
    let signed = scalar.is_signed_integer();
    let operation = match (operator, signed) {
        (CheckedBinaryOperator::Add, true) => "sadd",
        (CheckedBinaryOperator::Add, false) => "uadd",
        (CheckedBinaryOperator::Subtract, true) => "ssub",
        (CheckedBinaryOperator::Subtract, false) => "usub",
        (CheckedBinaryOperator::Multiply, true) => "smul",
        (CheckedBinaryOperator::Multiply, false) => "umul",
    };
    format!("llvm.{operation}.with.overflow.i{}", llvm_width(scalar))
}

fn checked_binary_intrinsic_signature(scalar: ScalarType) -> (&'static str, &'static str) {
    match llvm_width(scalar) {
        8 => ("{ i8, i1 }", "i8, i8"),
        16 => ("{ i16, i1 }", "i16, i16"),
        32 => ("{ i32, i1 }", "i32, i32"),
        64 => ("{ i64, i1 }", "i64, i64"),
        128 => ("{ i128, i1 }", "i128, i128"),
        _ => unreachable!("KIR verification accepted a non-integer checked width"),
    }
}

fn compare_predicate(predicate: ComparePredicate, ty: &Type) -> &'static str {
    let scalar = ty.as_scalar().expect("validated scalar comparison");
    if scalar == ScalarType::F32 {
        return match predicate {
            ComparePredicate::Equal => "oeq",
            ComparePredicate::NotEqual => "une",
            ComparePredicate::LessThan => "olt",
            ComparePredicate::LessThanOrEqual => "ole",
            ComparePredicate::GreaterThan => "ogt",
            ComparePredicate::GreaterThanOrEqual => "oge",
        };
    }
    let signed = scalar.is_signed_integer();
    match predicate {
        ComparePredicate::Equal => "eq",
        ComparePredicate::NotEqual => "ne",
        ComparePredicate::LessThan if signed => "slt",
        ComparePredicate::LessThan => "ult",
        ComparePredicate::LessThanOrEqual if signed => "sle",
        ComparePredicate::LessThanOrEqual => "ule",
        ComparePredicate::GreaterThan if signed => "sgt",
        ComparePredicate::GreaterThan => "ugt",
        ComparePredicate::GreaterThanOrEqual if signed => "sge",
        ComparePredicate::GreaterThanOrEqual => "uge",
    }
}

fn compare_opcode(ty: &Type) -> &'static str {
    if ty.as_scalar() == Some(ScalarType::F32) {
        "fcmp"
    } else {
        "icmp"
    }
}

fn cast_opcode(kind: CastKind, from: &Type, to: &Type) -> &'static str {
    match kind {
        CastKind::Truncate => "trunc",
        CastKind::ZeroExtend => "zext",
        CastKind::SignExtend => "sext",
        CastKind::IntegerToFloat if from.as_scalar().is_some_and(ScalarType::is_signed_integer) => {
            "sitofp"
        }
        CastKind::IntegerToFloat => "uitofp",
        CastKind::FloatToInteger if to.as_scalar().is_some_and(ScalarType::is_signed_integer) => {
            "fptosi"
        }
        CastKind::FloatToInteger => "fptoui",
        CastKind::Bitcast => "bitcast",
        CastKind::FloatExtend | CastKind::FloatTruncate => {
            unreachable!("preflight rejected unsupported cast")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_symbols_are_intentionally_conservative() {
        for symbol in ["fill", "_fill_2", "Fill42"] {
            assert!(is_safe_symbol(symbol), "{symbol}");
        }
        for symbol in ["", "42fill", "fill.kernel", "fill-kernel", "fill\nret void"] {
            assert!(!is_safe_symbol(symbol), "{symbol}");
        }
    }

    #[test]
    fn constants_and_signedness_are_stable() {
        assert_eq!(constant_value(&Constant::I32(-7)).unwrap(), "-7");
        assert_eq!(
            constant_value(&Constant::F32Bits(1.0f32.to_bits())).unwrap(),
            "0x3FF0000000000000"
        );
        assert_eq!(
            compare_predicate(ComparePredicate::LessThan, &Type::Scalar(ScalarType::I32)),
            "slt"
        );
        assert_eq!(
            compare_predicate(ComparePredicate::LessThan, &Type::INDEX),
            "ult"
        );
        assert_eq!(
            compare_predicate(
                ComparePredicate::GreaterThan,
                &Type::Scalar(ScalarType::F32)
            ),
            "ogt"
        );
        assert_eq!(
            compare_predicate(ComparePredicate::NotEqual, &Type::Scalar(ScalarType::F32)),
            "une"
        );
        assert_eq!(compare_opcode(&Type::Scalar(ScalarType::F32)), "fcmp");
        assert_eq!(compare_opcode(&Type::INDEX), "icmp");
    }

    #[test]
    fn equal_width_semantic_integer_bitcasts_are_supported() {
        let u64_ty = Type::Scalar(ScalarType::U64);
        let index_ty = Type::Scalar(ScalarType::Index);

        assert!(
            validate_cast(
                CastKind::Bitcast,
                &u64_ty,
                &index_ty,
                LoweringTarget::Gfx942XnackMinusV1
            )
            .is_ok()
        );
        assert!(
            validate_cast(
                CastKind::Bitcast,
                &index_ty,
                &u64_ty,
                LoweringTarget::Gfx942XnackMinusV1
            )
            .is_ok()
        );
    }

    #[test]
    fn ordinary_scalar_arithmetic_is_selected_by_type_not_workload_name() {
        let target = LoweringTarget::Gfx942XnackMinusV1;
        assert!(supported_binary(BinaryOp::Divide, &Type::INDEX, target));
        assert!(supported_binary(BinaryOp::Remainder, &Type::INDEX, target));
        assert!(supported_binary(
            BinaryOp::BitAnd,
            &Type::Scalar(ScalarType::U32),
            target
        ));
        assert!(supported_binary(
            BinaryOp::ShiftLeft,
            &Type::Scalar(ScalarType::U32),
            target
        ));
        assert!(supported_binary(BinaryOp::Divide, &Type::F32, target));
        assert!(!supported_binary(
            BinaryOp::Divide,
            &Type::F32,
            LoweringTarget::Baseline
        ));
    }

    #[test]
    fn binary_support_matrix_is_explicit_for_every_scalar_and_operator() {
        let scalars = [
            ScalarType::Bool,
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::I128,
            ScalarType::U8,
            ScalarType::U16,
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::U128,
            ScalarType::Index,
            ScalarType::F16,
            ScalarType::Bf16,
            ScalarType::F32,
            ScalarType::F64,
        ];
        let operators = [
            BinaryOp::Add,
            BinaryOp::Subtract,
            BinaryOp::Multiply,
            BinaryOp::Divide,
            BinaryOp::Remainder,
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::ShiftLeft,
            BinaryOp::ShiftRight,
            BinaryOp::Checked(CheckedBinaryOperator::Add),
            BinaryOp::Checked(CheckedBinaryOperator::Subtract),
            BinaryOp::Checked(CheckedBinaryOperator::Multiply),
        ];
        let ordinary_integer = |scalar| {
            matches!(
                scalar,
                ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::U8
                    | ScalarType::U16
                    | ScalarType::U32
                    | ScalarType::U64
                    | ScalarType::Index
            )
        };

        for target in [LoweringTarget::Baseline, LoweringTarget::Gfx942XnackMinusV1] {
            for scalar in scalars {
                for operator in operators {
                    let expected = match operator {
                        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => {
                            ordinary_integer(scalar) || scalar == ScalarType::F32
                        }
                        BinaryOp::Divide => {
                            ordinary_integer(scalar)
                                || (scalar == ScalarType::F32
                                    && !matches!(target, LoweringTarget::Baseline))
                        }
                        BinaryOp::Remainder => ordinary_integer(scalar),
                        BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                            scalar == ScalarType::Bool || ordinary_integer(scalar)
                        }
                        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => ordinary_integer(scalar),
                        BinaryOp::Checked(_) => scalar.is_integer(),
                    };
                    assert_eq!(
                        supported_binary(operator, &Type::Scalar(scalar), target),
                        expected,
                        "unexpected support for {operator:?} over {scalar:?} on {target:?}"
                    );
                }
            }
        }

        for (operator, opcode) in [
            (BinaryOp::BitAnd, "and"),
            (BinaryOp::BitOr, "or"),
            (BinaryOp::BitXor, "xor"),
        ] {
            assert_eq!(binary_opcode(operator, &Type::BOOL), opcode);
        }
        assert_eq!(
            binary_opcode(BinaryOp::ShiftLeft, &Type::Scalar(ScalarType::U32)),
            "shl"
        );
        assert_eq!(
            binary_opcode(BinaryOp::ShiftRight, &Type::Scalar(ScalarType::I32)),
            "ashr"
        );
        assert_eq!(
            binary_opcode(BinaryOp::ShiftRight, &Type::Scalar(ScalarType::U32)),
            "lshr"
        );
    }

    #[test]
    fn unary_support_matrix_distinguishes_logical_not_and_bitwise_complement() {
        let scalars = [
            ScalarType::Bool,
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::I128,
            ScalarType::U8,
            ScalarType::U16,
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::U128,
            ScalarType::Index,
            ScalarType::F16,
            ScalarType::Bf16,
            ScalarType::F32,
            ScalarType::F64,
        ];
        let ordinary_integer = |scalar| {
            matches!(
                scalar,
                ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::U8
                    | ScalarType::U16
                    | ScalarType::U32
                    | ScalarType::U64
                    | ScalarType::Index
            )
        };

        for scalar in scalars {
            let ty = Type::Scalar(scalar);
            let expected_negate = matches!(
                scalar,
                ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::F32
            );
            let expected_not = scalar == ScalarType::Bool || ordinary_integer(scalar);
            assert_eq!(
                supported_unary(UnaryOp::Negate, &ty),
                expected_negate,
                "unexpected Negate support for {scalar:?}"
            );
            assert_eq!(
                supported_unary(UnaryOp::Not, &ty),
                expected_not,
                "unexpected Not support for {scalar:?}"
            );
        }

        assert_eq!(
            unary_lowering_style(UnaryOp::Not, &Type::BOOL),
            Some(LlvmUnaryStyle::BooleanNot)
        );
        assert_eq!(
            unary_lowering_style(UnaryOp::Not, &Type::Scalar(ScalarType::I32)),
            Some(LlvmUnaryStyle::IntegerNot)
        );
    }
}
