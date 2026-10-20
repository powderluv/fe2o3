use super::*;

fn valid<'a>(path: &'a str, integer: &'a str) -> WrapperContractV1<'a> {
    WrapperContractV1 {
        item: true,
        core: true,
        generic_arguments: 0,
        mir_available: true,
        path,
        safe: true,
        rust_abi: true,
        variadic: false,
        input_count: 2,
        first: Some(integer),
        second: Some(integer),
        result: Some(integer),
        target_integer: true,
    }
}

#[test]
fn every_supported_primitive_wrapper_is_source_only() {
    for integer in [
        "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize",
    ] {
        for method in ["saturating_add", "saturating_sub"] {
            let path = format!("core::num::<impl {integer}>::{method}");
            assert!(authenticate_contract_v1(valid(&path, integer)));
        }
    }
}

#[test]
fn untrusted_same_names_methods_or_owners_do_not_get_the_exemption() {
    for path in [
        "user::saturating_sub",
        "user::num::<impl u32>::saturating_sub",
        "core::num::<impl i32>::saturating_sub",
        "core::num::<impl u32>::saturating_mul",
        "core::num::<impl u32>::saturating_add_signed",
        "core::num::<impl u32>::saturating_sub_unsigned",
        "core::num::<impl u32>::saturating_sub::shim",
        "saturating_sub",
    ] {
        assert!(!authenticate_contract_v1(valid(path, "u32")), "{path}");
    }
    let contract = valid("core::num::<impl u32>::saturating_sub", "u32");
    for altered in [
        WrapperContractV1 {
            core: false,
            ..contract
        },
        WrapperContractV1 {
            item: false,
            ..contract
        },
        WrapperContractV1 {
            mir_available: false,
            ..contract
        },
        WrapperContractV1 {
            generic_arguments: 1,
            ..contract
        },
    ] {
        assert!(!authenticate_contract_v1(altered));
    }
}

#[test]
fn unsupported_types_and_mismatched_signature_are_closed() {
    for integer in ["i128", "u128", "f32", "f64", "bool", "&u32", "*mut u32"] {
        let path = format!("core::num::<impl {integer}>::saturating_add");
        assert!(!authenticate_contract_v1(valid(&path, integer)));
    }
    let contract = valid("core::num::<impl u32>::saturating_add", "u32");
    for altered in [
        WrapperContractV1 {
            safe: false,
            ..contract
        },
        WrapperContractV1 {
            rust_abi: false,
            ..contract
        },
        WrapperContractV1 {
            variadic: true,
            ..contract
        },
        WrapperContractV1 {
            input_count: 1,
            ..contract
        },
        WrapperContractV1 {
            first: None,
            ..contract
        },
        WrapperContractV1 {
            second: Some("i32"),
            ..contract
        },
        WrapperContractV1 {
            result: Some("u64"),
            ..contract
        },
        WrapperContractV1 {
            target_integer: false,
            ..contract
        },
    ] {
        assert!(!authenticate_contract_v1(altered));
    }
}

fn is_saturating_source_exception(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Binary(expr) if matches!(expr.op, syn::BinOp::Or(_)) => {
            is_saturating_source_exception(&expr.left)
                || is_saturating_source_exception(&expr.right)
        }
        syn::Expr::Call(call) => {
            let syn::Expr::Path(function) = call.func.as_ref() else {
                return false;
            };
            function.qself.is_none()
                && function.path.leading_colon.is_none()
                && function.path.segments.len() == 2
                && function
                    .path
                    .segments
                    .iter()
                    .zip([
                        "trusted",
                        "authenticate_reviewed_safe_core_saturating_integer_helper_v1",
                    ])
                    .all(|(segment, name)| {
                        segment.ident == name
                            && matches!(segment.arguments, syn::PathArguments::None)
                    })
                && call.args.len() == 2
                && call
                    .args
                    .iter()
                    .zip(["tcx", "instance"])
                    .all(|(arg, name)| {
                        matches!(arg, syn::Expr::Path(path)
                        if path.qself.is_none() && path.path.is_ident(name))
                    })
        }
        _ => false,
    }
}

fn returns_authentication_decision(block: &syn::Block, expected: bool) -> bool {
    let [syn::Stmt::Expr(syn::Expr::Return(result), Some(_))] = block.stmts.as_slice() else {
        return false;
    };
    let Some(syn::Expr::Call(result)) = result.expr.as_deref() else {
        return false;
    };
    matches!(result.func.as_ref(), syn::Expr::Path(path)
        if path.qself.is_none() && path.path.is_ident("Ok"))
        && result.args.len() == 1
        && matches!(result.args.first(), Some(syn::Expr::Lit(value))
            if matches!(&value.lit, syn::Lit::Bool(value) if value.value == expected))
}

#[test]
fn exemption_does_not_claim_a_compiler_terminal_or_body_elision() {
    let source = syn::parse_file(include_str!("../collector/inlined_source_safety_v1.rs"))
        .expect("shared source-safety policy must parse");
    let policy = source
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "reviewed_external_source" => {
                Some(function)
            }
            _ => None,
        })
        .expect("shared external-source policy");
    let local_guard = policy
        .block
        .stmts
        .iter()
        .find_map(|stmt| match stmt {
            syn::Stmt::Expr(syn::Expr::If(branch), _) => Some(branch),
            _ => None,
        })
        .expect("local owners must be excluded before source exceptions");
    let syn::Expr::Binary(condition) = local_guard.cond.as_ref() else {
        panic!("expected local-owner or unsupported-instance refusal");
    };
    assert!(matches!(condition.op, syn::BinOp::Or(_)));
    let syn::Expr::MethodCall(local) = condition.left.as_ref() else {
        panic!("expected local-owner check");
    };
    assert!(local.method == "is_local" && local.args.is_empty());
    let syn::Expr::MethodCall(definition) = local.receiver.as_ref() else {
        panic!("expected the original instance definition");
    };
    assert!(definition.method == "def_id" && definition.args.is_empty());
    assert!(matches!(definition.receiver.as_ref(), syn::Expr::Path(path)
        if path.qself.is_none() && path.path.is_ident("instance")));
    assert!(returns_authentication_decision(
        &local_guard.then_branch,
        false
    ));
    let admission = policy
        .block
        .stmts
        .iter()
        .find_map(|stmt| match stmt {
            syn::Stmt::Expr(syn::Expr::If(branch), _)
                if is_saturating_source_exception(&branch.cond) =>
            {
                Some(branch)
            }
            _ => None,
        })
        .expect("saturation must remain an external-source exception");
    assert!(returns_authentication_decision(
        &admission.then_branch,
        true
    ));
    let terminal = include_str!("../production_semantic_terminal_v1.rs");
    assert!(!terminal.contains("authenticate_reviewed_safe_core_saturating_integer_helper_v1"));
}

#[test]
fn source_exception_guard_requires_the_actual_call_and_original_arguments() {
    let call =
        "trusted::authenticate_reviewed_safe_core_saturating_integer_helper_v1(tcx, instance)";
    for expression in [call.to_owned(), format!("other || {call} || atomic")] {
        assert!(is_saturating_source_exception(
            &syn::parse_str(&expression).unwrap()
        ));
    }
    for expression in [
        format!("/* {call} */ false"),
        format!("\"{call}\""),
        call.replace("trusted::", "untrusted::"),
        call.replace("(tcx, instance)", "(tcx, replacement)"),
        call.replace("(tcx, instance)", "(tcx)"),
        format!("!{call}"),
        format!("false && {call}"),
    ] {
        assert!(
            !is_saturating_source_exception(&syn::parse_str(&expression).unwrap()),
            "{expression}"
        );
    }
}
