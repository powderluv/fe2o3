//! Closed compiler-module helper participation, not Rust/source authenticity.
//! The immutable verified module supplies exact call edges and launch context.
//! No caller-provided full-wave flag or type layout constructs this context.
use super::*;
use fe2o3_kernel_analysis::{AnalysisReport, Diagnostic, UnsupportedReason};

pub(super) struct Context<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    physical_root: Option<&'a fe2o3_kernel_analysis::UniformityPhysicalLaunchV2<'a>>,
    functions: Vec<&'a Function>,
    reports: Vec<AnalysisReport>,
    effectful: Vec<bool>,
}

fn refusal(module: &Module, detail: &'static str) -> LoweringErrors {
    LoweringErrors::one(
        LoweringLocation::module(module),
        LoweringDiagnosticCode::UnprovenBarrierConvergence,
        detail,
    )
}

impl<'a> Context<'a> {
    /// An inert LLVM construction contract. Source/ABI correspondence remains
    /// independently required by the ordinary Rust compiler pipeline.
    pub(super) fn check(
        module: &'a Module,
        target: LoweringTarget,
    ) -> Result<Option<Self>, LoweringErrors> {
        Self::check_with_physical_v1(module, target, None)
    }

    // Compose only the actual checked physical root with this same immutable
    // closed helper graph. Helpers still receive no ambient launch analysis.
    pub(super) fn check_with_physical_v1(
        module: &'a Module,
        target: LoweringTarget,
        physical: Option<&'a CompilerPhysicalLaunchV2<'a>>,
    ) -> Result<Option<Self>, LoweringErrors> {
        if let Some(physical) = physical {
            physical.check(module, target)?;
        }
        let has_helper_shuffle = module.functions.iter().any(|function| {
            function.role == FunctionRole::InternalHelper
                && function.body.as_ref().is_some_and(|body| {
                    body.blocks.iter().any(|block| {
                        block.operations.iter().any(|operation| {
                            matches!(
                                operation.kind,
                                OperationKind::Wave(WaveOperation {
                                    kind: WaveOperationKind::ShuffleIndex { .. },
                                    ..
                                })
                            )
                        })
                    })
                })
        });
        if !has_helper_shuffle || target != LoweringTarget::Gfx950XnackMinusV1 {
            return Ok(None);
        }
        verify_module(module).map_err(LoweringErrors::verification)?;
        if module.kernels.len() != 1 || module.functions.len() > MAX_COMPILER_MODULE_GRAPH_FUNCTIONS
        {
            return Err(refusal(
                module,
                "Wave64 helpers require one bounded actual kernel context",
            ));
        }
        let kernel = &module.kernels[0];
        if kernel.workgroup_size != Some(WorkgroupSize { x: 64, y: 1, z: 1 }) {
            return Err(refusal(
                module,
                "Wave64 helpers require exact workgroup [64,1,1]",
            ));
        }
        let root = module
            .functions
            .iter()
            .find(|function| function.id == kernel.entry)
            .filter(|function| {
                function.role == FunctionRole::KernelEntry && function.body.is_some()
            })
            .ok_or_else(|| refusal(module, "Wave64 helper actual kernel entry is absent"))?;
        let physical_root = physical
            .map(|physical| physical.entry(kernel))
            .transpose()?
            .flatten();
        if physical_root.is_some_and(|physical| {
            !std::ptr::eq(physical.module(), module)
                || !std::ptr::eq(physical.kernel(), kernel)
                || !std::ptr::eq(physical.function(), root)
        }) {
            return Err(physical_launch_v2::invalid(module));
        }
        let mut trap_declaration = None;
        for function in &module.functions {
            if std::ptr::eq(function, root) || function.role == FunctionRole::InternalHelper {
                continue;
            }
            let exact_trap = function.role == FunctionRole::ExternalImport
                && function.body.is_none()
                && function.signature.parameters.is_empty()
                && function.signature.results.is_empty()
                && function.required_capabilities.len() == 1
                && matches!(function.required_capabilities.iter().next(),
                    Some(TargetCapability::Extension { namespace, name })
                    if namespace == AMDGPU_DIAGNOSTICS_CAPABILITY_NAMESPACE && name == AMDGPU_DIAGNOSTICS_CAPABILITY_NAME)
                && matches!(
                    AmdGpuDiagnosticOperation::from_intrinsic_call(&function.id, &[]),
                    Some(AmdGpuDiagnosticOperation::Trap)
                );
            if !exact_trap || trap_declaration.replace(function).is_some() {
                return Err(refusal(
                    module,
                    "Wave64 helper context has an additional definition or declaration",
                ));
            }
        }
        let mut functions = vec![root];
        functions.extend(
            module
                .functions
                .iter()
                .filter(|function| function.role == FunctionRole::InternalHelper),
        );
        let mut index = BTreeMap::new();
        for (position, function) in functions.iter().enumerate() {
            if index.insert(function.id.clone(), position).is_some() {
                return Err(refusal(
                    module,
                    "Wave64 helper function identities are ambiguous",
                ));
            }
            if position != 0
                && (function.body.is_none()
                    || function
                        .signature
                        .parameters
                        .iter()
                        .chain(&function.signature.results)
                        .any(|ty| !matches!(ty, Type::Scalar(_))))
            {
                return Err(refusal(
                    module,
                    "Wave64 helper transport is not an actual scalar function ABI",
                ));
            }
        }
        let mut edges = vec![BTreeSet::new(); functions.len()];
        let mut local_wave = vec![false; functions.len()];
        let mut rows = 0usize;
        let mut root_trap_called = false;
        for (position, function) in functions.iter().enumerate() {
            let body = function
                .body
                .as_ref()
                .ok_or_else(|| refusal(module, "Wave64 helper body is absent"))?;
            for block in &body.blocks {
                if position != 0
                    && (block
                        .parameters
                        .iter()
                        .any(|value| !matches!(value.ty, Type::Scalar(_)))
                        || block
                            .operations
                            .iter()
                            .flat_map(|operation| &operation.results)
                            .any(|value| !matches!(value.ty, Type::Scalar(_))))
                {
                    return Err(refusal(
                        module,
                        "Wave64 helper body has nonscalar transport",
                    ));
                }
                rows = block
                    .operations
                    .len()
                    .checked_add(1)
                    .and_then(|count| rows.checked_add(count))
                    .ok_or_else(|| refusal(module, "Wave64 helper graph accounting overflow"))?;
                if rows > MAX_COMPILER_MODULE_CALL_EDGES {
                    return Err(refusal(
                        module,
                        "Wave64 helper graph exceeds the existing compiler-module graph envelope",
                    ));
                }
                for operation in &block.operations {
                    match &operation.kind {
                        OperationKind::Call { callee, arguments } => {
                            if let Some(callee_index) = index.get(callee).copied() {
                                if callee_index == 0 {
                                    return Err(refusal(
                                        module,
                                        "Wave64 helpers cannot call a kernel entry",
                                    ));
                                }
                                edges[position].insert(callee_index);
                            } else if position == 0
                                && operation.results.is_empty()
                                && arguments.is_empty()
                                && trap_declaration
                                    .is_some_and(|declaration| &declaration.id == callee)
                                && matches!(
                                    AmdGpuDiagnosticOperation::from_intrinsic_call(
                                        callee, arguments
                                    ),
                                    Some(AmdGpuDiagnosticOperation::Trap)
                                )
                            {
                                // Actual KIR verification above requires final Trap + Unreachable.
                                root_trap_called = true;
                            } else {
                                return Err(refusal(
                                    module,
                                    "Wave64 helper graph contains an unavailable callee",
                                ));
                            }
                        }
                        OperationKind::Wave(wave) if position != 0 => {
                            if wave.width != WaveWidth::Wave64
                                || wave.active_lanes != 64
                                || !matches!(
                                    wave.kind,
                                    WaveOperationKind::LaneId
                                        | WaveOperationKind::ShuffleIndex { tile_width: 64, .. }
                                )
                                || wave.convergence.scope() != SynchronizationScope::Subgroup
                            {
                                return Err(refusal(
                                    module,
                                    "Wave64 helper operation has a different participation contract",
                                ));
                            }
                            local_wave[position] = true;
                        }
                        OperationKind::Constant(_)
                        | OperationKind::Unary { .. }
                        | OperationKind::Binary { .. }
                        | OperationKind::Compare { .. }
                        | OperationKind::Cast { .. }
                        | OperationKind::Select { .. } => {}
                        _ if position == 0 => {}
                        _ => {
                            return Err(refusal(
                                module,
                                "Wave64 helpers must remain scalar-pure apart from checked waves and defined calls",
                            ));
                        }
                    }
                }
            }
        }
        if trap_declaration.is_some() != root_trap_called {
            return Err(refusal(
                module,
                "Wave64 helper Trap declaration/call roster differs",
            ));
        }
        // Exact rooted DAG: no recursion, unused helper authority, or unknown edge.
        let mut colors = vec![0u8; functions.len()];
        let mut postorder = Vec::with_capacity(functions.len());
        let mut pending = vec![(0usize, false)];
        while let Some((node, exit)) = pending.pop() {
            if exit {
                colors[node] = 2;
                postorder.push(node);
                continue;
            }
            match colors[node] {
                2 => continue,
                1 => {
                    return Err(refusal(
                        module,
                        "Wave64 helper recursion has no participation contract",
                    ));
                }
                _ => {}
            }
            colors[node] = 1;
            pending.push((node, true));
            for callee in edges[node].iter().rev() {
                pending.push((*callee, false));
            }
        }
        if colors.iter().any(|color| *color != 2) {
            return Err(refusal(
                module,
                "Wave64 helper is not reachable from the actual kernel entry",
            ));
        }
        let mut effectful = local_wave;
        for node in postorder {
            let inherited = edges[node].iter().any(|callee| effectful[*callee]);
            effectful[node] = effectful[node] || inherited;
        }
        let mut reports = Vec::with_capacity(functions.len());
        for (position, function) in functions.iter().enumerate() {
            let report = if position == 0 {
                match physical_root {
                    Some(physical) => physical.analyze(),
                    None => fe2o3_kernel_analysis::analyze_kernel_entry(module, function),
                }
            } else {
                fe2o3_kernel_analysis::analyze_module_helper_v1(module, function)
                    .ok_or_else(|| refusal(module, "Wave64 helper analysis owner differs"))?
            };
            reports.push(report);
        }
        let context = Self {
            module,
            kernel,
            physical_root,
            functions,
            reports,
            effectful,
        };
        for (position, function) in context.functions.iter().enumerate() {
            let report = &context.reports[position];
            for diagnostic in report.diagnostics() {
                match diagnostic {
                    Diagnostic::Unsupported { .. }
                        if context.admits_call_diagnostic(function, diagnostic) => {}
                    _ => {
                        let mut location = if position == 0 {
                            LoweringLocation::function(module, kernel, function)
                        } else {
                            LoweringLocation::device_function(module, function)
                        };
                        if let Diagnostic::Unsupported {
                            block,
                            operation_index,
                            ..
                        } = diagnostic
                        {
                            location.block = *block;
                            location.operation = *operation_index;
                        }
                        return Err(LoweringErrors::one(
                            location,
                            LoweringDiagnosticCode::UnprovenBarrierConvergence,
                            format!(
                                "Wave64 helper convergence analysis has an undischarged diagnostic: {diagnostic:?}"
                            ),
                        ));
                    }
                }
            }
            for block in &function.body.as_ref().expect("checked actual body").blocks {
                for (operation_index, operation) in block.operations.iter().enumerate() {
                    let needs_uniform = match &operation.kind {
                        OperationKind::Wave(_) => true,
                        OperationKind::Call { callee, .. } => index
                            .get(callee)
                            .is_some_and(|callee| context.effectful[*callee]),
                        _ => false,
                    };
                    if needs_uniform
                        && !report
                            .block_control(block.id)
                            .is_uniform_for(SynchronizationScope::Subgroup)
                    {
                        let location = if position == 0 {
                            LoweringLocation::operation(
                                module,
                                kernel,
                                function,
                                block.id,
                                operation_index,
                            )
                        } else {
                            LoweringLocation::device_operation(
                                module,
                                function,
                                block.id,
                                operation_index,
                            )
                        };
                        return Err(LoweringErrors::one(
                            location,
                            LoweringDiagnosticCode::UnprovenBarrierConvergence,
                            format!(
                                "Wave64 operation or transitive helper call has varying control: {:?}",
                                report.block_control(block.id)
                            ),
                        ));
                    }
                }
            }
        }
        Ok(Some(context))
    }

    // Recheck the exact optional root input at final preflight. A helper
    // cannot borrow this physical root, even within the same module.
    pub(super) fn matches_physical_input_v1(
        &self,
        module: &Module,
        function: &Function,
        physical: Option<&fe2o3_kernel_analysis::UniformityPhysicalLaunchV2<'_>>,
    ) -> bool {
        if !std::ptr::eq(self.module, module) {
            return false;
        }
        match self.position(function) {
            Some(0) => match (self.physical_root, physical) {
                (None, None) => true,
                (Some(expected), Some(actual)) => std::ptr::eq(expected, actual),
                _ => false,
            },
            Some(_) => physical.is_none(),
            None => false,
        }
    }

    pub(super) fn function_is_convergent(&self, module: &Module, function: &Function) -> bool {
        std::ptr::eq(self.module, module)
            && self
                .position(function)
                .is_some_and(|position| self.effectful[position])
    }
    pub(super) fn call_is_convergent(
        &self,
        module: &Module,
        function: &Function,
        block: BlockId,
        operation_index: usize,
        callee: &FunctionId,
    ) -> bool {
        if !std::ptr::eq(self.module, module) || self.position(function).is_none() {
            return false;
        }
        let actual = function
            .body
            .as_ref()
            .and_then(|body| body.blocks.iter().find(|candidate| candidate.id == block))
            .and_then(|block| block.operations.get(operation_index));
        matches!(actual.map(|operation| &operation.kind), Some(OperationKind::Call { callee: actual, .. }) if actual == callee)
            && self
                .functions
                .iter()
                .enumerate()
                .skip(1)
                .any(|(position, function)| &function.id == callee && self.effectful[position])
    }
    fn position(&self, function: &Function) -> Option<usize> {
        self.functions
            .iter()
            .position(|member| std::ptr::eq(*member, function))
    }
    pub(super) fn helper_workgroup(
        &self,
        module: &Module,
        function: &Function,
    ) -> Option<WorkgroupSize> {
        if !std::ptr::eq(self.module, module) {
            return None;
        }
        let position = self.position(function)?;
        (position != 0 && self.effectful[position]).then_some(self.kernel.workgroup_size?)
    }
    pub(super) fn report(&self, module: &Module, function: &Function) -> Option<&AnalysisReport> {
        if !std::ptr::eq(self.module, module) {
            return None;
        }
        self.reports.get(self.position(function)?)
    }
    pub(super) fn admits_call_diagnostic(
        &self,
        function: &Function,
        diagnostic: &Diagnostic,
    ) -> bool {
        let Some(position) = self.position(function) else {
            return false;
        };
        let Diagnostic::Unsupported {
            block: Some(block),
            operation_index: Some(operation),
            reason: UnsupportedReason::CallWithoutSummary { callee },
        } = diagnostic
        else {
            return false;
        };
        let actual = self.functions[position]
            .body
            .as_ref()
            .and_then(|body| body.blocks.iter().find(|candidate| candidate.id == *block))
            .and_then(|block| block.operations.get(*operation));
        matches!(actual.map(|operation| &operation.kind), Some(OperationKind::Call { callee: actual, .. }) if actual == callee)
            && self
                .functions
                .iter()
                .skip(1)
                .any(|function| &function.id == callee)
    }
}

pub(super) fn owner_mismatch(module: &Module) -> LoweringErrors {
    refusal(module, "Wave64 helper convergence context owner differs")
}
#[cfg(test)]
#[path = "wave64_helper_context_v1_tests.rs"]
mod tests;
