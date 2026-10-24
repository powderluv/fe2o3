#![cfg(not(feature = "kernel-simt-gemm-general"))]

use fe2o3_device::{DisjointSlice, Index1D, KernelMarkerV1, Tiled2D};
use fe2o3_tiled_gemm_general_v1::{
    GENERAL_TILED_GEMM_PROTECTED_EXECUTION_BLOCKER_V1,
    GENERAL_TILED_GEMM_PROTECTED_EXECUTION_SUPPORTED_V1,
    GENERAL_TILED_GEMM_QUALIFICATION_EXECUTION_SUPPORTED_V1,
    GENERAL_TILED_GEMM_SAFE_SOURCE_PRESENT_V1, GENERAL_TILED_GEMM_SOURCE_LOWERING_SUPPORTED_V1,
    GENERAL_TILED_GEMM_SOURCE_TO_IR_SUPPORTED_V1,
    kernel::{__fe2o3_kernel_marker_tiled_gemm_general_v1, GENERAL_TILED_GEMM_WORKGROUP_V1},
};
use syn::visit::Visit;

const LIB_SOURCE: &str = include_str!("../src/lib.rs");
const KERNEL_SOURCE: &str = include_str!("../src/kernel.rs");

type GeneralKernelFn = fn(
    &[u16],
    &[u16],
    DisjointSlice<f32, Tiled2D<Index1D, 64, 16, 16, 4>>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    f32,
);

#[derive(Default)]
struct SourceFacts {
    unsafe_blocks: usize,
    unsafe_functions: usize,
    loop_depth: usize,
    try_expressions: usize,
    loop_try_expressions: usize,
    loop_return_expressions: usize,
    function_calls: Vec<String>,
    method_calls: Vec<String>,
    indexed_paths: Vec<String>,
}

impl<'ast> Visit<'ast> for SourceFacts {
    fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
        self.unsafe_blocks += 1;
        syn::visit::visit_expr_unsafe(self, expression);
    }

    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        if function.sig.unsafety.is_some() {
            self.unsafe_functions += 1;
        }
        syn::visit::visit_item_fn(self, function);
    }

    fn visit_expr_while(&mut self, expression: &'ast syn::ExprWhile) {
        self.loop_depth += 1;
        syn::visit::visit_expr_while(self, expression);
        self.loop_depth -= 1;
    }

    fn visit_expr_try(&mut self, expression: &'ast syn::ExprTry) {
        self.try_expressions += 1;
        if self.loop_depth != 0 {
            self.loop_try_expressions += 1;
        }
        syn::visit::visit_expr_try(self, expression);
    }

    fn visit_expr_return(&mut self, expression: &'ast syn::ExprReturn) {
        if self.loop_depth != 0 {
            self.loop_return_expressions += 1;
        }
        syn::visit::visit_expr_return(self, expression);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref()
            && let Some(segment) = path.path.segments.last()
        {
            self.function_calls.push(segment.ident.to_string());
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.method_calls.push(call.method.to_string());
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_index(&mut self, expression: &'ast syn::ExprIndex) {
        if let syn::Expr::Path(path) = expression.expr.as_ref()
            && let Some(identifier) = path.path.get_ident()
        {
            self.indexed_paths.push(identifier.to_string());
        }
        syn::visit::visit_expr_index(self, expression);
    }
}

#[test]
fn attributed_safe_kernel_compiles_with_the_dynamic_abi() {
    let function: GeneralKernelFn =
        <__fe2o3_kernel_marker_tiled_gemm_general_v1 as KernelMarkerV1>::FUNCTION;
    let _: GeneralKernelFn = function;
    let _: core::marker::PhantomData<
        fe2o3_tiled_gemm_general_v1::kernel::tiled_gemm_general_v1_gpu::Marker,
    > = core::marker::PhantomData;
}

#[test]
fn source_forbids_unsafe_and_contains_matrix_tiling_and_epilogue() {
    let library = syn::parse_file(LIB_SOURCE).expect("library source parses");
    assert!(library.attrs.iter().any(|attribute| {
        attribute.path().is_ident("forbid")
            && matches!(
                &attribute.meta,
                syn::Meta::List(list) if list.tokens.to_string().contains("unsafe_code")
            )
    }));

    let syntax = syn::parse_file(KERNEL_SOURCE).expect("kernel source parses");
    let mut facts = SourceFacts::default();
    facts.visit_file(&syntax);
    assert_eq!(facts.unsafe_blocks, 0);
    assert_eq!(facts.unsafe_functions, 0);
    assert_eq!(
        facts.try_expressions, 0,
        "checked view errors use let-else; paired_contract verifies both exact failure branches"
    );
    assert_eq!(
        facts.loop_try_expressions, 0,
        "lane-local checked loads must not return early from the MFMA loop"
    );
    assert_eq!(
        facts.loop_return_expressions, 0,
        "every lane entering the K loop must reconverge before MFMA"
    );
    assert!(facts.function_calls.iter().any(|call| call == "index_1d"));
    for required in [
        "Bf16MfmaAMatrix::row_major",
        "Bf16MfmaBMatrix::row_major",
        "F32AccumulatorFragment::zero",
        "WorkgroupPipeline::<Bf16MfmaAFragment<'_>, 2, 64, 1>::current",
        "WorkgroupPipeline::<Bf16MfmaBFragment<'_>, 2, 64, 1>::current",
        "let phase_count = (k as usize + 15) / 16",
        "while phase_index < phase_count",
        "let future_epoch = phase_index + 1",
        "alpha * values[0] + beta * *output",
    ] {
        assert!(KERNEL_SOURCE.contains(required), "missing `{required}`");
    }
    for required in [
        "checked_tiled_2d",
        "load_m16k16",
        "load_k16n16",
        "multiply_accumulate",
        "into_values",
        "get_tiled_2d_mut",
        "stage",
        "write",
        "commit",
        "wait",
        "consume",
        "read",
        "discard",
        "release",
    ] {
        assert!(
            facts.method_calls.iter().any(|call| call == required),
            "missing method `{required}`"
        );
    }
    for forbidden in ["get_unchecked", "get_unchecked_mut", "get_mut_at"] {
        assert!(!facts.method_calls.iter().any(|call| call == forbidden));
    }
    assert!(
        !facts.method_calls.iter().any(|call| call == "ok_or"),
        "checked values must reconverge instead of becoming early returns"
    );
    for input in ["a", "b"] {
        assert!(
            !facts.indexed_paths.iter().any(|path| path == input),
            "input `{input}` must not use MIR-asserting slice indexing"
        );
    }
}

#[test]
fn ordinary_host_execution_panics_before_output_mutation() {
    let function: GeneralKernelFn =
        <__fe2o3_kernel_marker_tiled_gemm_general_v1 as KernelMarkerV1>::FUNCTION;
    let a = [0x3f80_u16; 17];
    let b = [0x3f80_u16; 19];
    let sentinel = f32::from_bits(0x7f7f_ffff);
    let mut output = [sentinel; 19];
    // SAFETY: the test exclusively owns `output` for the caught invocation.
    let output_view = unsafe {
        DisjointSlice::<f32, Tiled2D<Index1D, 64, 16, 16, 4>>::from_raw_parts(
            output.as_mut_ptr(),
            output.len(),
        )
    };
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        function(&a, &b, output_view, 1, 1, 17, 17, 1, 19, 2.0, -1.0);
    }));
    assert!(failure.is_err());
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() == sentinel.to_bits())
    );
}

#[test]
fn status_records_current_fail_closed_boundaries() {
    assert_eq!(GENERAL_TILED_GEMM_WORKGROUP_V1, [64, 1, 1]);
    assert!(std::hint::black_box(
        GENERAL_TILED_GEMM_SAFE_SOURCE_PRESENT_V1
    ));
    assert!(std::hint::black_box(
        GENERAL_TILED_GEMM_SOURCE_TO_IR_SUPPORTED_V1
    ));
    assert!(std::hint::black_box(
        GENERAL_TILED_GEMM_SOURCE_LOWERING_SUPPORTED_V1
    ));
    assert!(std::hint::black_box(
        GENERAL_TILED_GEMM_QUALIFICATION_EXECUTION_SUPPORTED_V1
    ));
    assert!(!std::hint::black_box(
        GENERAL_TILED_GEMM_PROTECTED_EXECUTION_SUPPORTED_V1
    ));
    assert_eq!(
        GENERAL_TILED_GEMM_PROTECTED_EXECUTION_BLOCKER_V1,
        "protected Worker publication and artifact-currentness admission remain separate from the qualification runner"
    );
}
