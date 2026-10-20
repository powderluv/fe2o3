//! Caller-derived closure origins on the collector's exact direct-call graph.

use super::{CollectedFunction, CollectedFunctionRole, CollectionResult};
use crate::closure_profile_v1::{
    BoundedClosureAdmissionV2, ClosureForwardSourceV1, ClosureOriginPolicyV1, ClosureOriginV1,
    ClosureProfileErrorV1, is_shim_receiver_call_v1, normalized_ty, observe_raw_closures_v1,
    resolve_direct_call,
};
use crate::device_ffi::{DeviceFfiInstanceIdentity, stable_instance_identity};
use crate::rustc_semantic_adapter_v1::rustc_mir_body_sha256_v1;
use crate::rustc_semantic_plan_v1::{SourceClosureWorkV1, source_signature_v1};
use rustc_abi::ExternAbi;
use rustc_middle::mir::{BasicBlock, Body, Local, TerminatorKind};
use rustc_middle::ty::{Instance, TyCtxt, TyKind};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

type Error = ClosureProfileErrorV1;
pub(super) type CallGraphV1 = BTreeMap<
    DeviceFfiInstanceIdentity,
    BTreeMap<DeviceFfiInstanceIdentity, BTreeMap<BasicBlock, Box<[usize]>>>,
>;

pub(super) struct AuthenticatedClosureFlowV1<'tcx> {
    functions: Box<[(Instance<'tcx>, CollectedFunctionRole)]>,
    bodies: Box<[[u8; 32]]>,
    graph: CallGraphV1,
    work: SourceClosureWorkV1,
    references: super::reference_custody_v1::RetainedReferenceInputsV1<'tcx>,
}

fn charge(work: &mut SourceClosureWorkV1, amount: usize) -> Result<(), Error> {
    work.charge(amount)
        .map_err(|error| Error::new(error.to_string()))
}

pub(super) fn record_call_v1<'tcx>(
    graph: &mut CallGraphV1,
    tcx: TyCtxt<'tcx>,
    (caller, callee): (Instance<'tcx>, Instance<'tcx>),
    body: &Body<'tcx>,
    block: BasicBlock,
    work: &mut SourceClosureWorkV1,
) -> Result<(), Error> {
    charge(work, 1)?;
    let data = body
        .basic_blocks
        .get(block)
        .ok_or_else(|| Error::new("direct-call occurrence block is absent"))?;
    let TerminatorKind::Call { args, .. } = &data.terminator().kind else {
        return Err(Error::new("direct-call occurrence is not a call"));
    };
    let mut ordinals = Vec::new();
    for (ordinal, argument) in args.iter().enumerate() {
        charge(work, 1)?;
        let mut ty = normalized_ty(tcx, caller, argument.node.ty(body, tcx), "call operand")?;
        while let TyKind::Ref(_, pointee, _) = ty.kind() {
            charge(work, 1)?;
            ty = *pointee;
        }
        if matches!(ty.kind(), TyKind::Closure(..)) {
            charge(work, 1)?;
            ordinals.push(ordinal);
        }
    }
    if graph
        .entry(stable_instance_identity(tcx, caller))
        .or_default()
        .entry(stable_instance_identity(tcx, callee))
        .or_default()
        .insert(block, ordinals.into_boxed_slice())
        .is_some()
    {
        return Err(Error::new("duplicate direct-call occurrence"));
    }
    Ok(())
}

pub(super) fn authenticate_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &mut [CollectedFunction<'tcx>],
    graph: CallGraphV1,
    work: SourceClosureWorkV1,
    verbose: bool,
) -> Result<AuthenticatedClosureFlowV1<'tcx>, Error> {
    authenticate_with_enrollment_v1(tcx, functions, graph, work, verbose, None, None)
}

pub(super) fn authenticate_with_enrollment_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &mut [CollectedFunction<'tcx>],
    graph: CallGraphV1,
    mut work: SourceClosureWorkV1,
    verbose: bool,
    enrollment: Option<super::reference_enrollment_v1::RetainedEnrollmentV1<'tcx>>,
    loan: Option<&crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>>,
) -> Result<AuthenticatedClosureFlowV1<'tcx>, Error> {
    let admissions = observe_v1(tcx, functions, &graph, &mut work)?;
    for (function, admission) in functions.iter_mut().zip(admissions) {
        if verbose && let Some(admission) = &admission {
            eprintln!(
                "[collector] bounded closure admission: {} environment(s), {} static invocation(s)",
                admission.environments().len(),
                admission.calls().len()
            );
        }
        function.closure_observation = admission.map(|value| Box::new(value.into_observation()));
    }
    charge(&mut work, functions.len())?;
    let references =
        super::reference_custody_v1::RetainedReferenceInputsV1::capture_with_enrollment(
            tcx, functions, &mut work, enrollment, loan,
        )
        .map_err(|error| Error::new(error.to_string()))?;
    Ok(AuthenticatedClosureFlowV1 {
        functions: functions.iter().map(|f| (f.instance, f.role)).collect(),
        bodies: functions
            .iter()
            .map(|f| rustc_mir_body_sha256_v1(tcx, f.instance))
            .collect(),
        graph,
        work,
        references,
    })
}

impl<'tcx> AuthenticatedClosureFlowV1<'tcx> {
    pub(super) fn rederive_reference_bindings_and_inventory_v1(
        &mut self,
        tcx: TyCtxt<'tcx>,
        collection: &CollectionResult<'tcx>,
        loan: Option<
            &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
        >,
    ) -> Result<
        (
            crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
            Option<super::reference_custody_v1::PendingOriginalRootAssociationsV1<'tcx>>,
        ),
        crate::reference_effect_v1::ReferenceBindingErrorV1,
    > {
        self.references.rederive_with_inventory_capture(
            tcx,
            &collection.functions,
            &mut self.work,
            loan,
        )
    }

    pub(super) fn rederive_reference_bindings_v1(
        &mut self,
        tcx: TyCtxt<'tcx>,
        collection: &CollectionResult<'tcx>,
    ) -> Result<
        crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
        crate::reference_effect_v1::ReferenceBindingErrorV1,
    > {
        self.rederive_reference_bindings_with_enrollment_v1(tcx, collection, None)
    }

    pub(super) fn rederive_reference_bindings_with_enrollment_v1(
        &mut self,
        tcx: TyCtxt<'tcx>,
        collection: &CollectionResult<'tcx>,
        loan: Option<
            &crate::protected_compiler_execution::native_v3::ReferenceEnrollmentLoanV1<'_>,
        >,
    ) -> Result<
        crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
        crate::reference_effect_v1::ReferenceBindingErrorV1,
    > {
        self.references
            .rederive_with_enrollment(tcx, &collection.functions, &mut self.work, loan)
    }

    pub(super) fn revalidate_for_import_v1(
        mut self,
        tcx: TyCtxt<'tcx>,
        collection: &CollectionResult<'tcx>,
    ) -> Result<SourceClosureWorkV1, Error> {
        charge(&mut self.work, collection.functions.len())?;
        if self.functions.len() != collection.functions.len()
            || !self
                .functions
                .iter()
                .copied()
                .eq(collection.functions.iter().map(|f| (f.instance, f.role)))
        {
            return Err(Error::new(
                "collected closure-flow instance or role roster changed",
            ));
        }
        let known = collection
            .functions
            .iter()
            .map(|f| stable_instance_identity(tcx, f.instance))
            .collect::<BTreeSet<_>>();
        let mut observed = CallGraphV1::new();
        if self.bodies.len() != collection.functions.len() {
            return Err(Error::new("collected closure-flow body roster changed"));
        }
        for (function, retained_body) in collection.functions.iter().zip(&self.bodies) {
            if rustc_mir_body_sha256_v1(tcx, function.instance) != *retained_body {
                return Err(Error::new("collected function MIR changed before import"));
            }
            let body = tcx.instance_mir(function.instance.def);
            for (block, data) in body.basic_blocks.iter_enumerated() {
                charge(&mut self.work, 1)?;
                let TerminatorKind::Call { func, .. } = &data.terminator().kind else {
                    continue;
                };
                let callee = resolve_direct_call(tcx, function.instance, func)?;
                if known.contains(&stable_instance_identity(tcx, callee)) {
                    record_call_v1(
                        &mut observed,
                        tcx,
                        (function.instance, callee),
                        body,
                        block,
                        &mut self.work,
                    )?;
                } else {
                    validate_boundary_v1(tcx, callee, collection, &mut self.work)?;
                }
            }
        }
        if observed != self.graph {
            return Err(Error::new(
                "collected direct-call occurrences changed before import",
            ));
        }
        // Rebuild from live MIR and external roles, never from retained origins.
        let admissions = observe_v1(tcx, &collection.functions, &observed, &mut self.work)?;
        for (function, admission) in collection.functions.iter().zip(admissions) {
            charge(&mut self.work, 1)?;
            let observation = admission.map(BoundedClosureAdmissionV2::into_observation);
            if observation.as_ref() != function.closure_observation.as_deref() {
                return Err(Error::new(
                    "collected closure observation changed before import",
                ));
            }
        }
        Ok(self.work)
    }
}

fn observe_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
    graph: &CallGraphV1,
    work: &mut SourceClosureWorkV1,
) -> Result<Vec<Option<BoundedClosureAdmissionV2>>, Error> {
    charge(work, functions.len())?;
    let identities = functions
        .iter()
        .enumerate()
        .map(|(index, f)| (stable_instance_identity(tcx, f.instance), index))
        .collect::<BTreeMap<_, _>>();
    if identities.len() != functions.len() {
        return Err(Error::new("duplicate collected function identity"));
    }
    let mut raw = Vec::with_capacity(functions.len());
    let mut call_targets = Vec::with_capacity(functions.len());
    let mut nodes = BTreeMap::new();
    let mut origins = Vec::new();
    for (index, function) in functions.iter().enumerate() {
        let mut forwarding = BTreeMap::<usize, BTreeSet<usize>>::new();
        let mut targets = BTreeMap::new();
        if let Some(callees) = graph.get(&stable_instance_identity(tcx, function.instance)) {
            for (callee, sites) in callees {
                charge(work, 1)?;
                let callee = *identities
                    .get(callee)
                    .ok_or_else(|| Error::new("call graph contains an uncollected callee"))?;
                for (block, ordinals) in sites {
                    charge(work, 1)?;
                    if targets.insert(block.as_usize(), callee).is_some() {
                        return Err(Error::new("one call occurrence has multiple callees"));
                    }
                    charge(work, ordinals.len())?;
                    let arguments = ordinals.iter().copied().collect::<BTreeSet<_>>();
                    if arguments.len() != ordinals.len() {
                        return Err(Error::new("duplicate closure call argument"));
                    }
                    forwarding.insert(block.as_usize(), arguments);
                }
            }
        }
        let observation = observe_raw_closures_v1(tcx, function.instance, &forwarding, work)?;
        for (block, ordinals) in &mut forwarding {
            charge(work, 1)?;
            if ordinals.contains(&0)
                && is_shim_receiver_call_v1(
                    tcx,
                    function.instance,
                    BasicBlock::from_usize(*block),
                    work,
                )?
            {
                ordinals.remove(&0);
            }
        }
        // Every incoming closure value must have provenance, including values
        // hidden in projections of otherwise unobserved aggregate locals.
        let consumed = observation.iter().flat_map(|observation| {
            observation
                .calls()
                .iter()
                .map(|call| (call.block, 0))
                .chain(
                    observation
                        .forwards()
                        .iter()
                        .map(|site| (site.block, site.argument)),
                )
        });
        consume_arguments_v1(&mut forwarding, consumed, work)?;
        if let Some(observation) = &observation {
            for environment in observation.environments() {
                charge(work, 1)?;
                nodes.insert((index, environment.local), origins.len());
                origins.push(match (environment.origin, function.role) {
                    (ClosureOriginV1::DeviceInternal, _) => LOCAL,
                    (_, CollectedFunctionRole::InternalHelper) => 0,
                    _ => EXTERNAL,
                });
            }
        }
        raw.push(observation);
        call_targets.push(targets);
    }
    charge(work, origins.len())?;
    let mut edges = vec![Vec::new(); origins.len()];
    for (caller, observation) in raw.iter().enumerate() {
        let Some(observation) = observation else {
            continue;
        };
        for forward in observation.forwards() {
            charge(work, 1)?;
            let callee = *call_targets[caller]
                .get(&forward.block)
                .ok_or_else(|| Error::new("closure forwarding has no exact collected call site"))?;
            let formal = validate_forward_v1(
                tcx,
                functions[caller].instance,
                functions[callee].instance,
                forward.block,
                forward.argument,
                work,
            )?;
            let destination = *nodes.get(&(callee, formal)).ok_or_else(|| {
                Error::new("closure forwarding formal is not an observed environment")
            })?;
            match &forward.source {
                ClosureForwardSourceV1::Local(local) => {
                    let source = *nodes.get(&(caller, *local)).ok_or_else(|| {
                        Error::new("closure forwarding source is not an observed environment")
                    })?;
                    edges[source].push(destination);
                }
                ClosureForwardSourceV1::EmptyConstant(constant) => {
                    constant.revalidate(
                        tcx,
                        functions[caller].instance,
                        forward.block,
                        forward.argument,
                        work,
                    )?;
                    charge(work, 1)?;
                    origins[destination] |= LOCAL;
                }
            }
        }
    }
    propagate_origins_v1(&mut origins, &edges, work)?;
    let mut resolved = vec![BTreeMap::new(); functions.len()];
    for ((function, local), node) in nodes {
        charge(work, 1)?;
        let origin = if origins[node] & EXTERNAL != 0 {
            ClosureOriginV1::HostArgument
        } else {
            ClosureOriginV1::DeviceInternal
        };
        resolved[function].insert(local, origin);
    }
    raw.into_iter()
        .enumerate()
        .map(|(index, observation)| {
            observation
                .map(|value| value.admit(&resolved[index], ClosureOriginPolicyV1::Either, work))
                .transpose()
        })
        .collect()
}

fn consume_arguments_v1(
    recorded: &mut BTreeMap<usize, BTreeSet<usize>>,
    consumed: impl IntoIterator<Item = (usize, usize)>,
    work: &mut SourceClosureWorkV1,
) -> Result<(), Error> {
    for (block, argument) in consumed {
        charge(work, 1)?;
        if !recorded
            .get_mut(&block)
            .is_some_and(|ordinals| ordinals.remove(&argument))
        {
            return Err(Error::new(
                "closure use has no unique recorded call argument",
            ));
        }
    }
    for ordinals in recorded.values() {
        charge(work, 1)?;
        if !ordinals.is_empty() {
            return Err(Error::new(
                "closure call argument has no validated provenance",
            ));
        }
    }
    Ok(())
}

fn validate_boundary_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    callee: Instance<'tcx>,
    collection: &CollectionResult<'tcx>,
    work: &mut SourceClosureWorkV1,
) -> Result<(), Error> {
    charge(work, 1)?;
    if crate::production_semantic_terminal_v1::classify(tcx, callee.def_id()).is_some()
        || crate::production_rustc_intrinsic_v1::classify(tcx, callee)
            .map_err(|error| Error::new(error.to_string()))?
            .is_some()
        || crate::production_safe_core_shift_v1::SafeCoreShiftV1::classify(tcx, callee)
            .map_err(Error::new)?
            .is_some()
    {
        return Ok(());
    }
    if crate::production_primitive_from_v1::check_primitive_from_v1(
        tcx,
        callee,
        crate::production_primitive_from_v1::PrimitiveFromStageV1::ClosureReplay,
        work.limits(),
        &mut |amount| work.charge(amount),
    )
    .map_err(|error| Error::new(error.to_string()))?
    .is_some()
    {
        return Ok(());
    }
    for import in &collection.device_ffi.imports {
        charge(work, 1)?;
        if crate::device_ffi::source_owner_matches_instance(tcx, &import.owner, callee) {
            let target = collection
                .device_ffi
                .target
                .as_deref()
                .ok_or_else(|| Error::new("device FFI import has no authenticated target"))?;
            let contract = crate::device_ffi::contract_for_instance(tcx, callee, target)
                .map_err(|error| Error::new(error.to_string()))?;
            if contract.as_ref() == Some(&import.contract)
                && import.contract.direction == crate::device_ffi::DeviceFfiDirection::Import
            {
                return Ok(());
            }
            return Err(Error::new(
                "device FFI import boundary changed before import",
            ));
        }
    }
    Err(Error::new(
        "direct call reaches an uncollected non-boundary function",
    ))
}

fn validate_forward_v1<'tcx>(
    tcx: TyCtxt<'tcx>,
    caller: Instance<'tcx>,
    callee: Instance<'tcx>,
    block: usize,
    argument: usize,
    work: &mut SourceClosureWorkV1,
) -> Result<usize, Error> {
    charge(work, 1)?;
    let caller_body = tcx.instance_mir(caller.def);
    let callee_body = tcx.instance_mir(callee.def);
    let signature = source_signature_v1(tcx, callee).map_err(Error::new)?;
    if signature.abi != ExternAbi::Rust
        || signature.c_variadic
        || callee_body.arg_count != signature.inputs().len()
    {
        return Err(Error::new(
            "closure forwarding requires ordinary Rust source arguments",
        ));
    }
    let formal = argument
        .checked_add(1)
        .ok_or_else(|| Error::new("closure formal index overflowed"))?;
    let declared = signature
        .inputs()
        .get(argument)
        .ok_or_else(|| Error::new("closure source argument is absent"))?;
    let local = callee_body
        .local_decls
        .get(Local::from_usize(formal))
        .ok_or_else(|| Error::new("closure MIR formal is absent"))?;
    let data = caller_body
        .basic_blocks
        .get(BasicBlock::from_usize(block))
        .ok_or_else(|| Error::new("closure forwarding block is absent"))?;
    let TerminatorKind::Call { func, args, .. } = &data.terminator().kind else {
        return Err(Error::new("closure forwarding occurrence is not a call"));
    };
    charge(work, 1)?;
    if resolve_direct_call(tcx, caller, func)? != callee || args.len() != signature.inputs().len() {
        return Err(Error::new(
            "closure forwarding callee or argument roster changed",
        ));
    }
    let operand = args
        .get(argument)
        .ok_or_else(|| Error::new("closure caller argument is absent"))?;
    let actual = normalized_ty(
        tcx,
        caller,
        operand.node.ty(caller_body, tcx),
        "forwarded closure operand",
    )?;
    let formal_ty = normalized_ty(tcx, callee, local.ty, "closure MIR formal")?;
    if actual != *declared || actual != formal_ty || !matches!(actual.kind(), TyKind::Closure(..)) {
        return Err(Error::new(
            "closure caller, signature and MIR formal types disagree",
        ));
    }
    Ok(formal)
}

const LOCAL: u8 = 1;
const EXTERNAL: u8 = 2;

fn propagate_origins_v1(
    origins: &mut [u8],
    edges: &[Vec<usize>],
    work: &mut SourceClosureWorkV1,
) -> Result<(), Error> {
    if origins.len() != edges.len() {
        return Err(Error::new("closure origin graph shape changed"));
    }
    charge(work, origins.len())?;
    let mut incoming = vec![0usize; origins.len()];
    for destinations in edges {
        for destination in destinations {
            charge(work, 1)?;
            let degree = incoming
                .get_mut(*destination)
                .ok_or_else(|| Error::new("closure edge destination is absent"))?;
            *degree = degree
                .checked_add(1)
                .ok_or_else(|| Error::new("closure indegree overflowed"))?;
        }
    }
    let mut pending = incoming
        .iter()
        .enumerate()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node))
        .collect::<VecDeque<_>>();
    let mut visited = 0;
    while let Some(node) = pending.pop_front() {
        charge(work, 1)?;
        visited += 1;
        if origins[node] == 0 || origins[node] & !(LOCAL | EXTERNAL) != 0 {
            return Err(Error::new("closure formal provenance is unresolved"));
        }
        for destination in &edges[node] {
            charge(work, 1)?;
            origins[*destination] |= origins[node];
            incoming[*destination] -= 1;
            if incoming[*destination] == 0 {
                pending.push_back(*destination);
            }
        }
    }
    if visited != origins.len() {
        return Err(Error::new("closure forwarding cycle is unsupported"));
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod primitive_from_stage_tests;
#[cfg(test)]
#[path = "closure_flow_v1_tests.rs"]
mod tests;
