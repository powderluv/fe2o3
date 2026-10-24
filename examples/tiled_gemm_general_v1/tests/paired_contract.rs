#![forbid(unsafe_code)]

//! Source/marker and independent host-reference checks, not device execution evidence.

use fe2o3_device::{DisjointSlice, Index1D, KernelMarkerV1};
use fe2o3_tiled_gemm_general_v1::reference::{
    ReferenceProblemV1, evaluate_reference_v1, strided_extent_v1,
};
use syn::visit::Visit;

const FEATURE: &str = "kernel-simt-gemm-general";
const PAD: f32 = f32::from_bits(0x7f80_0001);

fn problem() -> ReferenceProblemV1 {
    ReferenceProblemV1 {
        rows: 2,
        columns: 3,
        reduction: 2,
        lhs_stride: 3,
        rhs_stride: 4,
        output_stride: 5,
        product_scale: 2.0,
        output_scale: -0.5,
    }
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "physical slot {index}"
        );
    }
}

#[test]
fn selected_marker_retains_the_exact_source_abi() {
    #[cfg(not(feature = "kernel-simt-gemm-general"))]
    use fe2o3_tiled_gemm_general_v1::kernel::{
        __fe2o3_kernel_marker_tiled_gemm_general_v1 as Marker,
        GENERAL_TILED_GEMM_WORKGROUP_V1 as WORKGROUP,
        tiled_gemm_general_v1_gpu::Marker as LaunchMarker,
    };
    #[cfg(feature = "kernel-simt-gemm-general")]
    use fe2o3_tiled_gemm_general_v1::kernel_simt::{
        __fe2o3_kernel_marker_simt_gemm_general_v1 as Marker,
        GENERAL_SIMT_GEMM_WORKGROUP_V1 as WORKGROUP,
        simt_gemm_general_v1_gpu::Marker as LaunchMarker,
    };

    #[cfg(not(feature = "kernel-simt-gemm-general"))]
    type OutputIndex = fe2o3_device::Tiled2D<Index1D, 64, 16, 16, 4>;
    #[cfg(feature = "kernel-simt-gemm-general")]
    type OutputIndex = Index1D;
    type KernelFn =
        fn(&[u16], &[u16], DisjointSlice<f32, OutputIndex>, u32, u32, u32, u32, u32, u32, f32, f32);
    let _: KernelFn = <Marker as KernelMarkerV1>::FUNCTION;
    let _: core::marker::PhantomData<LaunchMarker> = core::marker::PhantomData;
    let name = if cfg!(feature = "kernel-simt-gemm-general") {
        "simt_gemm_general_v1"
    } else {
        "tiled_gemm_general_v1"
    };
    assert_eq!(<Marker as KernelMarkerV1>::LOGICAL_NAME, name);
    assert_eq!(<Marker as KernelMarkerV1>::EXPORT_NAME, name);
    assert_eq!(WORKGROUP, [64, 1, 1]);
}

#[test]
fn feature_selects_exactly_one_source_module_on_every_target() {
    fn enabled(meta: &syn::Meta, selected: bool) -> bool {
        match meta {
            syn::Meta::NameValue(value) if value.path.is_ident("feature") => {
                let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(name),
                    ..
                }) = &value.value
                else {
                    panic!("feature value must be a string");
                };
                assert_eq!(name.value(), FEATURE);
                selected
            }
            syn::Meta::List(list) if list.path.is_ident("not") => {
                !enabled(&list.parse_args::<syn::Meta>().unwrap(), selected)
            }
            _ => panic!("source selection must depend only on its explicit feature"),
        }
    }
    let library = syn::parse_file(include_str!("../src/lib.rs")).unwrap();
    let modules = library
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(module) if module.ident == "kernel" || module.ident == "kernel_simt" => {
                Some(module)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 2);
    for selected in [false, true] {
        let active = modules
            .iter()
            .filter(|module| {
                let attributes = module
                    .attrs
                    .iter()
                    .filter(|attribute| attribute.path().is_ident("cfg"))
                    .collect::<Vec<_>>();
                assert_eq!(attributes.len(), 1);
                enabled(&attributes[0].parse_args::<syn::Meta>().unwrap(), selected)
            })
            .map(|module| module.ident.to_string())
            .collect::<Vec<_>>();
        assert_eq!(active, [if selected { "kernel_simt" } else { "kernel" }]);
    }
}

#[test]
fn both_variants_are_real_attributed_source_and_simt_uses_checked_scalar_accesses() {
    #[derive(Default)]
    struct Facts {
        calls: Vec<String>,
        methods: Vec<String>,
        paths: Vec<String>,
        unsafe_count: usize,
        indexing_count: usize,
        checked_view_let_else: usize,
    }
    impl<'ast> Visit<'ast> for Facts {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if let syn::Expr::Path(path) = call.func.as_ref() {
                self.calls.push(
                    path.path
                        .segments
                        .iter()
                        .map(|segment| segment.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::"),
                );
            }
            syn::visit::visit_expr_call(self, call);
        }
        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            self.methods.push(call.method.to_string());
            syn::visit::visit_expr_method_call(self, call);
        }
        fn visit_path(&mut self, path: &'ast syn::Path) {
            self.paths.extend(
                path.segments
                    .iter()
                    .map(|segment| segment.ident.to_string()),
            );
            syn::visit::visit_path(self, path);
        }
        fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
            self.unsafe_count += 1;
            syn::visit::visit_expr_unsafe(self, expression);
        }
        fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
            self.unsafe_count += usize::from(function.sig.unsafety.is_some());
            syn::visit::visit_item_fn(self, function);
        }
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let Some(init) = &local.init
                && let syn::Expr::Call(call) = init.expr.as_ref()
                && let syn::Expr::Path(path) = call.func.as_ref()
            {
                let name = path
                    .path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if matches!(
                    name.as_str(),
                    "Bf16MfmaAMatrix::row_major"
                        | "Bf16MfmaBMatrix::row_major"
                        | "StridedReadView2D::from_shared_slice"
                ) {
                    let syn::Pat::TupleStruct(pattern) = &local.pat else {
                        panic!("Ok pattern")
                    };
                    assert!(pattern.path.is_ident("Ok"));
                    assert_eq!(pattern.elems.len(), 1);
                    let syn::Pat::Ident(binding) = &pattern.elems[0] else {
                        panic!("Ok binding")
                    };
                    assert!(
                        binding.by_ref.is_none()
                            && binding.mutability.is_none()
                            && binding.subpat.is_none()
                    );
                    let (_, failure) = init.diverge.as_ref().expect("checked failure branch");
                    let syn::Expr::Block(failure) = failure.as_ref() else {
                        panic!("failure block")
                    };
                    assert!(failure.label.is_none());
                    assert_eq!(failure.block.stmts.len(), 1);
                    let syn::Stmt::Expr(syn::Expr::Return(returned), Some(_)) =
                        &failure.block.stmts[0]
                    else {
                        panic!("checked error return")
                    };
                    let syn::Expr::Call(returned) = returned.expr.as_deref().unwrap() else {
                        panic!("Err value")
                    };
                    let syn::Expr::Path(constructor) = returned.func.as_ref() else {
                        panic!("Err constructor")
                    };
                    assert!(constructor.path.is_ident("Err"));
                    assert_eq!(returned.args.len(), 1);
                    let syn::Expr::Path(error) = &returned.args[0] else {
                        panic!("error code")
                    };
                    assert_eq!(
                        error
                            .path
                            .segments
                            .iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::"),
                        "KernelError::InvalidArgument"
                    );
                    self.checked_view_let_else += 1;
                }
            }
            syn::visit::visit_local(self, local);
        }

        fn visit_expr_index(&mut self, expression: &'ast syn::ExprIndex) {
            self.indexing_count += 1;
            syn::visit::visit_expr_index(self, expression);
        }
    }
    for (source, name) in [
        (include_str!("../src/kernel.rs"), "tiled_gemm_general_v1"),
        (
            include_str!("../src/kernel_simt.rs"),
            "simt_gemm_general_v1",
        ),
    ] {
        let syntax = syn::parse_file(source).unwrap();
        let kernels = syntax
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Fn(function)
                    if function.attrs.iter().any(|a| a.path().is_ident("kernel")) =>
                {
                    Some(function)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].sig.ident, name);
        assert_eq!(kernels[0].sig.inputs.len(), 11);
        let mut facts = Facts::default();
        facts.visit_file(&syntax);
        assert_eq!(facts.unsafe_count, 0);
        assert_eq!(facts.checked_view_let_else, 2);
        if name == "simt_gemm_general_v1" {
            fn is_zero_check(expression: &syn::Expr, name: &str) -> bool {
                let syn::Expr::Binary(binary) = expression else {
                    return false;
                };
                matches!(binary.op, syn::BinOp::Eq(_))
                    && matches!(binary.left.as_ref(), syn::Expr::Path(path) if path.path.is_ident(name))
                    && matches!(binary.right.as_ref(), syn::Expr::Lit(lit)
                        if matches!(&lit.lit, syn::Lit::Int(value) if value.base10_parse::<u32>().ok() == Some(0)))
            }
            let statements = &kernels[0].block.stmts;
            let (guard_index, guard) = statements
                .iter()
                .enumerate()
                .find_map(|(index, statement)| match statement {
                    syn::Stmt::Expr(syn::Expr::If(branch), _)
                        if is_zero_check(&branch.cond, "ldc") =>
                    {
                        Some((index, branch))
                    }
                    _ => None,
                })
                .expect("explicit original stride zero guard");
            let syn::Stmt::Expr(syn::Expr::If(empty), _) = &statements[guard_index - 1] else {
                panic!("empty output exits before stride zero guard")
            };
            let syn::Expr::Binary(condition) = empty.cond.as_ref() else {
                panic!("empty shape condition")
            };
            assert!(matches!(condition.op, syn::BinOp::Or(_)));
            assert!(is_zero_check(&condition.left, "m") && is_zero_check(&condition.right, "n"));
            assert!(guard.else_branch.is_none() && empty.else_branch.is_none());
            assert_eq!(empty.then_branch.stmts.len(), 1);
            let syn::Stmt::Expr(syn::Expr::Return(returned), Some(_)) = &empty.then_branch.stmts[0]
            else {
                panic!("empty shape return")
            };
            let syn::Expr::Call(ok) = returned.expr.as_deref().unwrap() else {
                panic!("empty success")
            };
            assert!(matches!(ok.func.as_ref(), syn::Expr::Path(path) if path.path.is_ident("Ok")));
            assert_eq!(ok.args.len(), 1);
            assert!(matches!(&ok.args[0], syn::Expr::Tuple(tuple) if tuple.elems.is_empty()));
            assert_eq!(guard.then_branch.stmts.len(), 1);
            let syn::Stmt::Expr(syn::Expr::Return(returned), Some(_)) = &guard.then_branch.stmts[0]
            else {
                panic!("zero stride refusal")
            };
            let syn::Expr::Call(err) = returned.expr.as_deref().unwrap() else {
                panic!("stride error")
            };
            assert!(
                matches!(err.func.as_ref(), syn::Expr::Path(path) if path.path.is_ident("Err"))
            );
            assert_eq!(err.args.len(), 1);
            let syn::Expr::Path(code) = &err.args[0] else {
                panic!("stride error code")
            };
            assert_eq!(
                code.path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::"),
                "KernelError::InvalidArgument"
            );
            for name in ["row", "column"] {
                let index = statements.iter().position(|statement| matches!(statement,
                    syn::Stmt::Local(local)
                        if matches!(&local.pat, syn::Pat::Ident(binding) if binding.ident == name)
                )).unwrap();
                assert!(
                    guard_index < index,
                    "stride guard precedes both division and remainder"
                );
            }
            assert_eq!(facts.indexing_count, 0);
            assert!(facts.calls.iter().any(|call| call == "thread::index_1d"));
            for call in ["StridedReadView2D::from_shared_slice", "Bf16::from_bits"] {
                assert_eq!(facts.calls.iter().filter(|found| *found == call).count(), 2);
            }
            for method in ["load_or", "to_f32", "get_mut"] {
                assert!(facts.methods.iter().any(|found| found == method));
            }
            for path in facts.paths {
                assert!(!path.starts_with("Bf16Mfma"));
                assert!(!matches!(
                    path.as_str(),
                    "Matrix" | "WorkgroupPipeline" | "WorkgroupLdsScope"
                ));
            }
            for method in [
                "get_unchecked",
                "get_unchecked_mut",
                "get_mut_at",
                "multiply_accumulate",
            ] {
                assert!(!facts.methods.iter().any(|found| found == method));
            }
        }
    }
}

#[test]
fn reference_matches_hand_computed_strided_alpha_beta_outputs() {
    let lhs = [0x3fc0, 0xc000, 0x7f81, 0xbf00, 0x4080];
    let rhs = [0x4000, 0xbf80, 0x3f00, 0x7f81, 0xc040, 0x4000, 0x3fc0];
    let initial = [2.0, 4.0, 6.0, PAD, -0.0, -2.0, 0.0, 8.0, PAD, -0.0];
    let actual = evaluate_reference_v1(&lhs, &rhs, &initial, problem()).unwrap();
    assert_bits(
        &actual,
        &[17.0, -13.0, -7.5, PAD, -0.0, -25.0, 17.0, 7.5, PAD, -0.0],
    );
}

#[test]
fn reference_handles_tile_tails_multiple_k_phases_and_exact_extents() {
    let problem = ReferenceProblemV1 {
        rows: 17,
        columns: 19,
        reduction: 23,
        lhs_stride: 27,
        rhs_stride: 24,
        output_stride: 23,
        product_scale: 0.75,
        output_scale: -0.25,
    };
    assert_eq!(strided_extent_v1(17, 23, 27), Some(455));
    assert_eq!(strided_extent_v1(23, 19, 24), Some(547));
    assert_eq!(strided_extent_v1(17, 19, 23), Some(387));
    let mut lhs = vec![0x7f81; 455];
    let mut rhs = vec![0x7f81; 547];
    for row in 0..17 {
        lhs[row * 27..row * 27 + 23].fill(0x3fc0);
    }
    for row in 0..23 {
        rhs[row * 24..row * 24 + 19].fill(0xc000);
    }
    for trailing in [0, 7] {
        let mut initial = vec![PAD; 387 + trailing];
        let mut expected = initial.clone();
        for row in 0..17 {
            initial[row * 23..row * 23 + 19].fill(4.0);
            expected[row * 23..row * 23 + 19].fill(-52.75);
        }
        let actual = evaluate_reference_v1(&lhs, &rhs, &initial, problem).unwrap();
        assert!(
            actual.iter().enumerate().all(|(index, value)| {
                index / 23 >= 17 || index % 23 >= 19 || value.is_finite()
            })
        );
        assert_bits(&actual, &expected);
    }
}

#[test]
fn reference_tail_oracle_varies_every_matrix_axis() {
    let problem = ReferenceProblemV1 {
        rows: 17,
        columns: 19,
        reduction: 23,
        lhs_stride: 27,
        rhs_stride: 24,
        output_stride: 23,
        product_scale: 0.5,
        output_scale: -0.25,
    };
    let mut lhs = vec![0x7f81; 455];
    let mut rhs = vec![0x7f81; 547];
    // These small integers are exactly representable BF16 values. Encoding
    // directly keeps this oracle independent of Bf16::from_f32 rounding.
    let integer_bits = |value: i32| ((value as f32).to_bits() >> 16) as u16;
    for row in 0..17 {
        for depth in 0..23 {
            lhs[row * 27 + depth] = integer_bits((row + 1 + depth) as i32);
        }
    }
    for depth in 0..23 {
        for column in 0..19 {
            rhs[depth * 24 + column] = integer_bits(2 * (column + 1) as i32 - depth as i32);
        }
    }
    let mut initial = vec![PAD; 394];
    let mut expected = initial.clone();
    for row in 0..17 {
        for column in 0..19 {
            let index = row * 23 + column;
            initial[index] = 8.0;
            // sum((a+d)*(b-d), d=0..22) uses sum(d)=253,
            // sum(d*d)=3795, not another matrix multiply implementation.
            let a = (row + 1) as i32;
            let b = 2 * (column + 1) as i32;
            let dot = 23 * a * b + (b - a) * 253 - 3795;
            expected[index] = 0.5 * dot as f32 - 2.0;
        }
    }
    let actual = evaluate_reference_v1(&lhs, &rhs, &initial, problem).unwrap();
    assert_bits(&actual, &expected);
    // Anchor independently hand-computed corners across both tile tails.
    assert_eq!(actual[0], -1750.0);
    assert_eq!(actual[16 * 23 + 18], 8186.0);
}

#[test]
fn bf16_raw_fractional_operands_keep_f32_product_and_accumulation_precision() {
    // Raw BF16 operands, not f32 inputs rounded by the test. The expected
    // binary32 bits come from exact rational products with denominator 2^14.
    let cases: &[(&[u16], &[u16], u32)] = &[
        (&[0x3f81], &[0x3f7f], 0x3f80_7f00),
        (&[0x3f81], &[0x3f81], 0x3f82_0200),
        (&[0x3f81, 0x3f7f], &[0x3f81, 0xbf80], 0x3ca0_8000),
        // 64.5^2 - 64*65 = 0.25. Rounding the products to BF16 gives zero,
        // whose error exceeds the unchanged general GEMM 2e-3 tolerance.
        (&[0x4281, 0x4280], &[0x4281, 0xc282], 0x3e80_0000),
    ];
    for &(lhs, rhs, expected_bits) in cases {
        let actual = evaluate_reference_v1(
            lhs,
            rhs,
            &[0.0, PAD],
            ReferenceProblemV1 {
                rows: 1,
                columns: 1,
                reduction: lhs.len() as u32,
                lhs_stride: lhs.len() as u32,
                rhs_stride: 1,
                output_stride: 1,
                product_scale: 1.0,
                output_scale: 0.0,
            },
        )
        .unwrap();
        assert_bits(&actual, &[f32::from_bits(expected_bits), PAD]);
    }
}

#[test]
fn bf16_fractional_m16n16k16_has_independent_f32_outputs_and_canaries() {
    let problem = ReferenceProblemV1 {
        rows: 16,
        columns: 16,
        reduction: 16,
        lhs_stride: 16,
        rhs_stride: 16,
        output_stride: 16,
        product_scale: 1.0,
        output_scale: 0.0,
    };
    let mut lhs = vec![0_u16; 256];
    let mut rhs = vec![0_u16; 256];
    for row in 0..16 {
        for depth in 0..16 {
            // A=(128+row+depth)/128, exactly encoded in BF16.
            lhs[row * 16 + depth] = 0x3f80 + (row + depth) as u16;
        }
    }
    for depth in 0..16 {
        for column in 0..16 {
            let numerator = 128 + 2 * column - depth;
            // B=(128+2*column-depth)/128 crosses the exponent boundary at1.
            rhs[depth * 16 + column] = if numerator >= 128 {
                0x3f80 + (numerator - 128) as u16
            } else {
                0x3f00 + 2 * (numerator - 64) as u16
            };
        }
    }
    let mut initial = vec![PAD; 264];
    initial[..256].fill(0.0);
    let mut expected = initial.clone();
    for row in 0..16 {
        for column in 0..16 {
            let a = (128 + row) as i32;
            let b = (128 + 2 * column) as i32;
            // sum((a+d)*(b-d),d=0..15)/128^2; sum(d)=120,sum(d*d)=1240.
            // All intermediates and the final binary32 result are exact here.
            expected[row * 16 + column] = (16 * a * b + 120 * (b - a) - 1240) as f32 / 16384.0;
        }
    }
    let actual = evaluate_reference_v1(&lhs, &rhs, &initial, problem).unwrap();
    assert!(actual[..256].iter().all(|value| value.is_finite()));
    assert_bits(&actual, &expected);
    assert_eq!(actual[0].to_bits(), 0x417e_ca00);
    assert_eq!(actual[255].to_bits(), 0x41b0_ca00);
}

#[test]
fn reference_zero_reduction_evaluates_the_complete_epilogue() {
    let problem = ReferenceProblemV1 {
        rows: 2,
        columns: 2,
        reduction: 0,
        lhs_stride: 0,
        rhs_stride: 0,
        output_stride: 3,
        product_scale: 7.0,
        output_scale: -0.5,
    };
    let initial = [2.0, -4.0, PAD, 6.0, 8.0, -0.0];
    let actual = evaluate_reference_v1(&[], &[], &initial, problem).unwrap();
    assert_bits(&actual, &[-1.0, 2.0, PAD, -3.0, -4.0, -0.0]);
    let nonfinite = evaluate_reference_v1(
        &[],
        &[],
        &initial,
        ReferenceProblemV1 {
            product_scale: f32::INFINITY,
            ..problem
        },
    )
    .unwrap();
    assert!(nonfinite[0].is_nan());
    assert_eq!(nonfinite[2].to_bits(), PAD.to_bits());
    let nan_initial = evaluate_reference_v1(
        &[],
        &[],
        &[PAD; 5],
        ReferenceProblemV1 {
            output_scale: 0.0,
            ..problem
        },
    )
    .unwrap();
    assert!(nan_initial[0].is_nan());
}

#[test]
fn reference_zero_output_still_validates_other_nonempty_operands() {
    let empty_rows = ReferenceProblemV1 {
        rows: 0,
        ..problem()
    };
    let empty_columns = ReferenceProblemV1 {
        columns: 0,
        ..problem()
    };
    let initial = [PAD, -0.0];
    assert_bits(
        &evaluate_reference_v1(&[], &[0; 7], &initial, empty_rows).unwrap(),
        &initial,
    );
    assert_bits(
        &evaluate_reference_v1(&[0; 5], &[], &initial, empty_columns).unwrap(),
        &initial,
    );
    for result in [
        evaluate_reference_v1(&[], &[0; 6], &initial, empty_rows),
        evaluate_reference_v1(&[0; 4], &[], &initial, empty_columns),
    ] {
        assert_eq!(
            result.unwrap_err(),
            "reference input is shorter than its declared strided extent"
        );
    }
}

#[test]
fn reference_rejects_invalid_strides_and_one_short_buffers() {
    let base = problem();
    for (problem, expected) in [
        (
            ReferenceProblemV1 {
                lhs_stride: 1,
                ..base
            },
            "lhs stride is smaller than the logical reduction extent",
        ),
        (
            ReferenceProblemV1 {
                rhs_stride: 2,
                ..base
            },
            "rhs stride is smaller than the logical column extent",
        ),
        (
            ReferenceProblemV1 {
                output_stride: 2,
                ..base
            },
            "output stride is smaller than the logical column extent",
        ),
        (
            ReferenceProblemV1 {
                lhs_stride: 0,
                rhs_stride: 0,
                output_stride: 0,
                ..base
            },
            "lhs stride is smaller than the logical reduction extent",
        ),
    ] {
        assert_eq!(
            evaluate_reference_v1(&[0; 5], &[0; 7], &[0.0; 8], problem).unwrap_err(),
            expected
        );
    }
    for (lhs, rhs, output) in [(4, 7, 8), (5, 6, 8), (5, 7, 7)] {
        assert_eq!(
            evaluate_reference_v1(&vec![0; lhs], &vec![0; rhs], &vec![0.0; output], base)
                .unwrap_err(),
            "reference input is shorter than its declared strided extent"
        );
    }
    let nan_product = evaluate_reference_v1(
        &[0x7f81],
        &[0x3f80],
        &[2.0],
        ReferenceProblemV1 {
            rows: 1,
            columns: 1,
            reduction: 1,
            lhs_stride: 1,
            rhs_stride: 1,
            output_stride: 1,
            product_scale: 0.0,
            output_scale: 1.0,
        },
    )
    .unwrap();
    assert!(nan_product[0].is_nan());
}
