use std::collections::BTreeMap;

use syn::{Expr, GenericArgument, ImplItem, Item, Pat, PathArguments, Stmt, Type};

const ENROLLMENT_ENTRY: &str = "verify_general_kernel_checks_with_enrollment_v1";
const ENTRIES: [&str; 9] = [
    "verify_general_kernel_checks",
    "lower_production_target",
    "export_simulation_bundle_v1",
    "export_simulation_bundle_v2",
    "export_simulation_bundle_v3",
    "export_simulation_bundle_v4",
    "export_simulation_bundle_v5",
    "export_simulation_bundle_v6",
    ENROLLMENT_ENTRY,
];
const COMMON: [&str; 5] = [
    "import_semantic_mir",
    "construct_semantic_middle_end",
    "construct_semantic_ssa",
    "materialize_target_neutral",
    "verify_general_kernel_checks",
];

fn collected_stage(ty: &Type) -> bool {
    let Type::Path(ty) = ty else { return false };
    if ty.qself.is_some() || ty.path.segments.len() != 1 {
        return false;
    }
    let outer = &ty.path.segments[0];
    let PathArguments::AngleBracketed(args) = &outer.arguments else {
        return false;
    };
    if outer.ident != "ProductionCompilation" || args.args.len() != 2 {
        return false;
    }
    let (GenericArgument::Lifetime(owner), GenericArgument::Type(Type::Path(stage))) =
        (&args.args[0], &args.args[1])
    else {
        return false;
    };
    if stage.qself.is_some() || stage.path.segments.len() != 1 {
        return false;
    }
    let stage = &stage.path.segments[0];
    let PathArguments::AngleBracketed(args) = &stage.arguments else {
        return false;
    };
    stage.ident == "CollectedRustStage"
        && args.args.len() == 1
        && matches!(&args.args[0], GenericArgument::Lifetime(inner) if inner.ident == owner.ident)
}

fn local_name(expr: &Expr) -> Option<String> {
    let Expr::Path(path) = expr else { return None };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return None;
    }
    path.path.get_ident().map(ToString::to_string)
}

fn exact_path(expr: &Expr, expected: &[&str]) -> bool {
    let Expr::Path(path) = expr else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == expected.len()
        && path
            .path
            .segments
            .iter()
            .zip(expected)
            .all(|(segment, name)| {
                segment.ident == *name && matches!(segment.arguments, PathArguments::None)
            })
}

// Follow receiver ownership through local bindings; textual mentions or calls
// disconnected from the returned stage cannot satisfy the route contract.
fn chain<'a>(
    expr: &'a Expr,
    locals: &mut BTreeMap<String, &'a Expr>,
    calls: &mut Vec<(&'a syn::ExprMethodCall, bool)>,
    propagated: bool,
) -> Result<(), &'static str> {
    match expr {
        Expr::Try(expr) if !propagated => chain(&expr.expr, locals, calls, true),
        Expr::MethodCall(call) if call.turbofish.is_none() => {
            chain(&call.receiver, locals, calls, false)?;
            calls.push((call, propagated));
            Ok(())
        }
        Expr::Path(_) if !propagated => {
            let name = local_name(expr).ok_or("nonlocal receiver")?;
            if name == "self" {
                Ok(())
            } else {
                let value = locals.remove(&name).ok_or("unknown or reused receiver")?;
                chain(value, locals, calls, false)
            }
        }
        _ => Err("not a checked receiver chain"),
    }
}

fn conditional_refusal(expr: &Expr) -> Option<&Expr> {
    let Expr::If(guard) = expr else { return None };
    let Expr::MethodCall(condition) = guard.cond.as_ref() else {
        return None;
    };
    if guard.else_branch.is_some()
        || condition.method != "has_direct_conditional_roots_v2"
        || condition.turbofish.is_some()
        || !condition.args.is_empty()
    {
        return None;
    }
    let [Stmt::Expr(Expr::Return(returned), Some(_))] = guard.then_branch.stmts.as_slice() else {
        return None;
    };
    let Expr::Call(error) = returned.expr.as_deref()? else {
        return None;
    };
    if local_name(&error.func).as_deref() != Some("Err") || error.args.len() != 1 {
        return None;
    }
    let Expr::MethodCall(refusal) = &error.args[0] else {
        return None;
    };
    (refusal.method == "conditional_production_finalizer_refusal_v5"
        && refusal.turbofish.is_none()
        && local_name(&refusal.receiver) == local_name(&condition.receiver)
        && local_name(&condition.receiver).is_some()
        && refusal.args.len() == 1
        && local_name(&refusal.args[0]).as_deref() == Some("target_budget"))
    .then_some(&condition.receiver)
}

fn check_entry(method: &syn::ImplItemFn) -> Result<(), &'static str> {
    let name = method.sig.ident.to_string();
    let mut expected = COMMON.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    if name == ENTRIES[0] {
        expected = vec![ENROLLMENT_ENTRY.into()];
    } else if name == ENROLLMENT_ENTRY {
        expected[0] = "import_semantic_mir_with_enrollment_v1".into();
    } else {
        expected.push("attach_target_neutral_checks".into());
        if name == ENTRIES[1] {
            expected.extend([
                "admit_formal_memory".into(),
                "lower_production_target".into(),
            ]);
        } else {
            expected.push(name.replacen("export_", "into_", 1));
        }
    }
    let Some((Stmt::Expr(tail, None), statements)) = method.block.stmts.split_last() else {
        return Err("missing returned endpoint");
    };
    let mut locals = BTreeMap::new();
    let mut guarded = false;
    for statement in statements {
        match statement {
            Stmt::Local(local) if !guarded => {
                let Pat::Ident(binding) = &local.pat else {
                    return Err("nonlocal binding");
                };
                let init = local.init.as_ref().ok_or("uninitialized stage")?;
                if binding.by_ref.is_some()
                    || binding.mutability.is_some()
                    || binding.subpat.is_some()
                    || init.diverge.is_some()
                    || locals
                        .insert(binding.ident.to_string(), init.expr.as_ref())
                        .is_some()
                {
                    return Err("ambiguous stage binding");
                }
            }
            Stmt::Expr(expr, None) if name == ENTRIES[1] && !guarded => {
                let receiver = conditional_refusal(expr).ok_or("changed conditional refusal")?;
                let mut guarded_calls = Vec::new();
                chain(receiver, &mut locals.clone(), &mut guarded_calls, false)?;
                if guarded_calls.len() != COMMON.len()
                    || guarded_calls
                        .iter()
                        .zip(COMMON)
                        .any(|((call, propagated), expected)| {
                            call.method != expected || !propagated || !call.args.is_empty()
                        })
                {
                    return Err("conditional refusal precedes ranked checks");
                }
                guarded = true;
            }
            _ => return Err("disconnected stage or alternate route"),
        }
    }
    if guarded != (name == ENTRIES[1]) {
        return Err("missing conditional refusal");
    }
    let mut calls = Vec::new();
    chain(tail, &mut locals, &mut calls, false)?;
    if !locals.is_empty() || calls.len() != expected.len() {
        return Err("unused or missing stage");
    }
    for (index, ((call, propagated), expected)) in calls.iter().zip(&expected).enumerate() {
        if call.method != expected.as_str() || *propagated != (index + 1 < calls.len()) {
            return Err("reordered, substituted, or unchecked stage");
        }
        if name == ENTRIES[0] {
            if call.args.len() != 1 || !exact_path(&call.args[0], &["None"]) {
                return Err("changed absent enrollment forwarding");
            }
        } else if name == ENROLLMENT_ENTRY && index == 0 {
            if call.args.len() != 2
                || !exact_path(
                    &call.args[0],
                    &["source_owned_v29", "ImportProfile", "Current"],
                )
                || !exact_path(&call.args[1], &["enrollment"])
            {
                return Err("changed enrollment import forwarding");
            }
        } else if expected.starts_with("into_simulation_bundle_v")
            && index + 1 == calls.len()
            && !expected.ends_with(['5', '6'])
        {
            if call.args.len() != 1
                || !exact_path(
                    &call.args[0],
                    &[
                        "fe2o3_kernel_ir",
                        "SimulationCompilerExecutionBindingV1",
                        "UnavailableExtractionOnly",
                    ],
                )
            {
                return Err("changed simulation binding");
            }
        } else if !call.args.is_empty() {
            return Err("unexpected stage arguments");
        }
    }
    Ok(())
}

fn check_routes(file: &syn::File) -> Result<(), String> {
    let mut entries = BTreeMap::new();
    for item in &file.items {
        let Item::Impl(block) = item else { continue };
        if block.trait_.is_some() || !collected_stage(&block.self_ty) {
            continue;
        }
        for item in &block.items {
            let ImplItem::Fn(method) = item else { continue };
            let name = method.sig.ident.to_string();
            if ENTRIES.contains(&name.as_str()) && entries.insert(name, method).is_some() {
                return Err("duplicate production entry".into());
            }
        }
    }
    if entries.len() != ENTRIES.len() {
        return Err("missing production entry".into());
    }
    for (name, method) in entries {
        check_entry(method).map_err(|error| format!("{name}: {error}"))?;
    }
    Ok(())
}

#[test]
fn every_production_entry_crosses_the_same_pre_ranked_materialization_stage() {
    check_routes(&syn::parse_file(include_str!("production_pipeline.rs")).unwrap()).unwrap();
}

fn replace_body(file: &mut syn::File, name: &str, body: &str) {
    let method = file
        .items
        .iter_mut()
        .filter_map(|item| match item {
            Item::Impl(block) if collected_stage(&block.self_ty) => Some(block),
            _ => None,
        })
        .flat_map(|block| &mut block.items)
        .find_map(|item| match item {
            ImplItem::Fn(method) if method.sig.ident == name => Some(method),
            _ => None,
        })
        .unwrap();
    method.block = syn::parse_str(body).unwrap();
}

#[test]
fn production_route_ast_check_rejects_changed_enrollment_forwarding() {
    let original = syn::parse_file(include_str!("production_pipeline.rs")).unwrap();
    check_routes(&original).unwrap();
    for body in [
        "{ self.verify_general_kernel_checks_with_enrollment_v1(Some(forged)) }",
        "{ self.verify_general_kernel_checks_with_enrollment_v1() }",
        "{ other.verify_general_kernel_checks_with_enrollment_v1(None) }",
        "{ self.verify_general_kernel_checks_with_enrollment_v1(None)? }",
    ] {
        let mut altered = original.clone();
        replace_body(&mut altered, ENTRIES[0], body);
        let error = check_routes(&altered).expect_err(body);
        assert!(error.starts_with(&format!("{}:", ENTRIES[0])), "{error}");
    }
    for import in [
        "import_semantic_mir()",
        "import_semantic_mir_with_enrollment_v1(source_owned_v29::ImportProfile::Current, None)",
        "import_semantic_mir_with_enrollment_v1(source_owned_v29::ImportProfile::NominalV35, enrollment)",
        "import_semantic_mir_with_enrollment_v1(source_owned_v29::ImportProfile::Current, replacement)",
    ] {
        let body = format!(
            "{{ self.{import}?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks() }}"
        );
        let mut altered = original.clone();
        replace_body(&mut altered, ENROLLMENT_ENTRY, &body);
        let error = check_routes(&altered).expect_err(&body);
        assert!(
            error.starts_with(&format!("{ENROLLMENT_ENTRY}:")),
            "{error}"
        );
    }
}

#[test]
fn production_route_ast_check_requires_one_enrollment_helper() {
    let original = syn::parse_file(include_str!("production_pipeline.rs")).unwrap();
    check_routes(&original).unwrap();
    for duplicate in [false, true] {
        let mut altered = original.clone();
        for item in &mut altered.items {
            let Item::Impl(block) = item else { continue };
            if !collected_stage(&block.self_ty) {
                continue;
            }
            if let Some(index) = block.items.iter().position(
                |item| matches!(item, ImplItem::Fn(method) if method.sig.ident == ENROLLMENT_ENTRY),
            ) {
                if duplicate {
                    block.items.push(block.items[index].clone());
                } else {
                    block.items.remove(index);
                }
                break;
            }
        }
        assert_eq!(
            check_routes(&altered).unwrap_err(),
            if duplicate {
                "duplicate production entry"
            } else {
                "missing production entry"
            },
        );
    }
}

#[test]
fn production_route_ast_check_rejects_missing_reordered_and_disconnected_gates() {
    let original = syn::parse_file(include_str!("production_pipeline.rs")).unwrap();
    check_routes(&original).unwrap();
    let mut inline = original.clone();
    replace_body(
        &mut inline,
        ENROLLMENT_ENTRY,
        "{ self.import_semantic_mir_with_enrollment_v1(source_owned_v29::ImportProfile::Current, enrollment)?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks() }",
    );
    check_routes(&inline).unwrap();
    for (body, reason) in [
        "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.verify_general_kernel_checks() }",
        "{ self.import_semantic_mir()?.construct_semantic_ssa()?.construct_semantic_middle_end()?.materialize_target_neutral()?.verify_general_kernel_checks() }",
        "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral().verify_general_kernel_checks() }",
        "{ let unused = self.materialize_target_neutral()?; other.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks() }",
        "{ let text = \".materialize_target_neutral()?\"; self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.verify_general_kernel_checks() }",
        "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.materialize_target_neutral()?.verify_general_kernel_checks() }",
    ].into_iter().zip([
        "unused or missing stage",
        "reordered, substituted, or unchecked stage",
        "reordered, substituted, or unchecked stage",
        "unknown or reused receiver",
        "unused or missing stage",
        "unused or missing stage",
    ]) {
        let mut altered = original.clone();
        let body = body.replace(
            "import_semantic_mir()",
            "import_semantic_mir_with_enrollment_v1(source_owned_v29::ImportProfile::Current, enrollment)",
        );
        replace_body(&mut altered, ENROLLMENT_ENTRY, &body);
        let error = check_routes(&altered).expect_err(&body);
        assert_eq!(error, format!("{ENROLLMENT_ENTRY}: {reason}"));
    }
    for (name, body) in [
        (
            ENTRIES[1],
            "{ let ranked = self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks()?; if ranked.has_direct_conditional_roots_v2() { return Err(ranked.conditional_production_finalizer_refusal_v5(target_budget)); } ranked.attach_target_neutral_checks()?.lower_production_target() }",
        ),
        (
            ENTRIES[1],
            "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks()?.attach_target_neutral_checks()?.admit_formal_memory()?.lower_production_target() }",
        ),
        (
            ENTRIES[2],
            "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks()?.attach_target_neutral_checks()?.into_simulation_bundle_v2(fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly) }",
        ),
        (
            ENTRIES[2],
            "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks()?.attach_target_neutral_checks()?.into_simulation_bundle_v1(forged_binding) }",
        ),
        (
            ENTRIES[7],
            "{ self.import_semantic_mir()?.construct_semantic_middle_end()?.construct_semantic_ssa()?.materialize_target_neutral()?.verify_general_kernel_checks()?.attach_target_neutral_checks()?.admit_formal_memory()?.into_simulation_bundle_v6() }",
        ),
    ] {
        let mut altered = original.clone();
        replace_body(&mut altered, name, body);
        let error = check_routes(&altered).expect_err(body);
        assert!(error.starts_with(&format!("{name}:")), "{error}");
    }
    let mut duplicate = original.clone();
    let routes = original
        .items
        .iter()
        .find(|item| {
            matches!(item,
        Item::Impl(block) if collected_stage(&block.self_ty) && block.items.iter().any(|item|
            matches!(item, ImplItem::Fn(method) if method.sig.ident == ENTRIES[0])))
        })
        .unwrap();
    duplicate.items.push(routes.clone());
    assert!(check_routes(&duplicate).is_err());
    let mut missing = original;
    missing.items.retain(|item| {
        !matches!(item,
        Item::Impl(block) if collected_stage(&block.self_ty) && block.items.iter().any(|item|
            matches!(item, ImplItem::Fn(method) if method.sig.ident == ENTRIES[0])))
    });
    assert!(check_routes(&missing).is_err());
}
