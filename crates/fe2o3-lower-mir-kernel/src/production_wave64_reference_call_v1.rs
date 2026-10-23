//! Borrowed source-SSA relation for the Wave64 arithmetic receiver and context.
//! This issues no EXEC, launch, convergence, target or artifact authority.

use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
};
use fe2o3_mir_model::semantic_mir_v1::*;
use fe2o3_mir_model::{SsaResolvedEventV1, SsaValueV1};
use fe2o3_pliron::{
    ProductionSemanticSsaEntryOriginV1 as EntryOrigin,
    ProductionSemanticSsaEventRoleV1 as EventRole,
    ProductionSemanticSsaFunctionOccurrencesV1 as Occurrences,
    ProductionSemanticSsaOccurrenceSiteV1 as Site,
    ProductionSemanticSsaOperandRoleV1 as OperandRole,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Debug)]
pub(super) enum ReferenceErrorV1 {
    Resource(Resource),
    Unavailable(&'static str),
    CallbackPanicked,
}
impl From<Resource> for ReferenceErrorV1 {
    fn from(error: Resource) -> Self {
        Self::Resource(error)
    }
}
type ResultV1<T> = Result<T, ReferenceErrorV1>;
fn refuse<T>(why: &'static str) -> ResultV1<T> {
    Err(ReferenceErrorV1::Unavailable(why))
}
fn unique<T>(slot: &mut Option<T>, value: T) -> ResultV1<()> {
    if slot.replace(value).is_some() {
        refuse("ambiguous original SSA occurrence")
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Context,
    LaneCarrier,
    LaneValue,
    Snapshot { lane_field: u32 },
    Zst,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceOrigin {
    RootIssuedSsa {
        function: SemanticFunctionIdV1,
        value: SsaValueV1,
    },
    RootRetainedCall {
        function: SemanticFunctionIdV1,
        block: SemanticBlockIdV1,
        local: SemanticLocalIdV1,
        context_type: SemanticTypeIdV1,
    },
    ZeroState,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Meaning {
    kind: Kind,
    ty: SemanticTypeIdV1,
    referent: Option<SemanticTypeIdV1>,
    origin: SourceOrigin,
}

/// Copyable coordinates are inspection data, never detached source authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ArgumentV1 {
    Snapshot {
        reference_type: SemanticTypeIdV1,
        snapshot_type: SemanticTypeIdV1,
        lane_field: u32,
    },
    Rank {
        scalar_type: SemanticTypeIdV1,
    },
    Context {
        reference_type: SemanticTypeIdV1,
        context_type: SemanticTypeIdV1,
    },
}
pub(super) struct CheckedReferenceCallV1<'a> {
    owner: &'a ProductionSemanticSsaOwnerV1,
    caller: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    helper: SemanticFunctionIdV1,
    arguments: [ArgumentV1; 2],
    meanings: [Meaning; 2],
}
impl CheckedReferenceCallV1<'_> {
    pub(super) fn owner(&self) -> &ProductionSemanticSsaOwnerV1 {
        self.owner
    }
    pub(super) fn caller(&self) -> SemanticFunctionIdV1 {
        self.caller
    }
    pub(super) fn block(&self) -> SemanticBlockIdV1 {
        self.block
    }
    pub(super) fn helper(&self) -> SemanticFunctionIdV1 {
        self.helper
    }
    pub(super) fn argument(&self, index: usize) -> Option<ArgumentV1> {
        self.arguments.get(index).copied()
    }
    pub(super) fn helper_declaration(&self) -> &SemanticFunctionDeclV1 {
        &self.owner.source_semantic().functions()[self.helper.index() as usize]
    }
    pub(super) fn source_call(&self) -> &SemanticDirectCallV1 {
        call_at(self.owner, self.caller, self.block).expect("checked immutable source call")
    }
    // Exact origins remain borrowed through this relation, not recreated from types.
    pub(super) fn same_source_origins(&self, other: &Self) -> bool {
        std::ptr::eq(self.owner, other.owner) && self.meanings == other.meanings
    }
}

fn call_at(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
) -> Option<&SemanticDirectCallV1> {
    match owner
        .source_semantic()
        .functions()
        .get(function.index() as usize)?
        .blocks()
        .get(block.index() as usize)?
        .terminator()
        .kind()
    {
        SemanticTerminatorKindV1::Call(call) => Some(call),
        _ => None,
    }
}
fn rows(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
) -> ResultV1<Occurrences<'_>> {
    owner
        .occurrences_v1()
        .and_then(|capture| capture.function(function))
        .ok_or(ReferenceErrorV1::Unavailable(
            "original source SSA occurrences required",
        ))
}
fn site(block: SemanticBlockIdV1) -> Site {
    Site::Terminator {
        block: fe2o3_mir_model::SsaBlockIdV1::new(block.index()),
    }
}
fn base_use(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    site: Site,
    role: OperandRole,
    budget: &mut Budget<'_>,
) -> ResultV1<SsaValueV1> {
    let mut found = None;
    for event in rows(owner, function)?.events() {
        budget.charge_work(1)?;
        if event.site() == site && event.operand() == role && event.role() == EventRole::BaseUse {
            let Some(SsaResolvedEventV1::Use { value, .. }) = event.resolved() else {
                return refuse("original operand has no SSA use");
            };
            if !event.is_reachable() || !event.is_promoted() {
                return refuse("original operand is unreachable or retained");
            }
            unique(&mut found, value)?;
        }
    }
    found.ok_or(ReferenceErrorV1::Unavailable(
        "original operand SSA occurrence absent",
    ))
}
fn result_value(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    budget: &mut Budget<'_>,
) -> ResultV1<Option<SsaValueV1>> {
    let call =
        call_at(owner, function, block).ok_or(ReferenceErrorV1::Unavailable("producer call"))?;
    let destination = call
        .destination()
        .ok_or(ReferenceErrorV1::Unavailable("producer destination"))?;
    if !destination.place().projections().is_empty()
        || !matches!(call.unwind(), SemanticUnwindActionV1::Unreachable)
    {
        return refuse("projected or exceptional producer destination");
    }
    let local = destination.place().local().index();
    let occurrences = rows(owner, function)?;
    let mut found = None;
    let mut retained = false;
    for edge in occurrences.edge_definitions() {
        budget.charge_work(1)?;
        if edge.edge().source().get() == block.index() && edge.variable().get() == local {
            if edge.edge().ordinal() != 0 || edge.ordinal() != 0 || !edge.is_reachable() {
                return refuse("unavailable original producer edge");
            }
            if edge.is_promoted() {
                if retained {
                    return refuse("mixed source producer storage classes");
                }
                unique(
                    &mut found,
                    edge.value().ok_or(ReferenceErrorV1::Unavailable(
                        "promoted producer lacks SSA value",
                    ))?,
                )?;
            } else {
                // An unrelated retained issuer is not an SSA candidate. Its
                // original call/local ancestry is checked separately at Borrow.
                if edge.value().is_some() || retained || found.is_some() {
                    return refuse("inconsistent retained original producer edge");
                }
                retained = true;
            }
        }
    }
    for event in occurrences.events() {
        budget.charge_work(1)?;
        if event.site() == site(block)
            && event.role() == EventRole::DestinationDefine
            && let Some(SsaResolvedEventV1::Define { variable, value }) = event.resolved()
            && variable.get() == local
        {
            if !event.is_reachable() || !event.is_promoted() {
                return refuse("unavailable producer definition");
            }
            unique(&mut found, value)?;
        }
    }
    Ok(found)
}

struct Resolver<'a> {
    owner: &'a ProductionSemanticSsaOwnerV1,
    root: SemanticFunctionIdV1,
    visits: usize,
}
include!("production_wave64_reference_call_v1/resolver_types.rs");
include!("production_wave64_reference_call_v1/retained_storage.rs");
include!("production_wave64_reference_call_v1/resolver_values.rs");

fn derive<'a>(
    owner: &'a ProductionSemanticSsaOwnerV1,
    caller: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    budget: &mut Budget<'_>,
) -> ResultV1<CheckedReferenceCallV1<'a>> {
    let source = owner.source_semantic();
    if source.roots().len() != 1 || source.functions().len() > 32 {
        return refuse("one actual root and bounded helper roster required");
    }
    let root = source.roots()[0];
    let root_decl = &source.functions()[root.index() as usize];
    if root_decl.role() != SemanticFunctionRoleV1::KernelRoot || root_decl.kernel_entry().is_none()
    {
        return refuse("original kernel root required");
    }
    let call = call_at(owner, caller, block)
        .ok_or(ReferenceErrorV1::Unavailable("original Defined call"))?;
    let Some(SemanticCallableDeclV1::Defined { function: helper }) =
        source.callables().get(call.callee().index() as usize)
    else {
        return refuse("reference transport requires actual Defined call");
    };
    let helper_decl = &source.functions()[helper.index() as usize];
    let abi = helper_decl.abi();
    if helper_decl.role() != SemanticFunctionRoleV1::InternalHelper
        || helper_decl.export().is_some()
        || call.arguments().len() != 3
        || abi.source_input_types().len() != 3
        || !call.variadic_argument_abis().is_empty()
        || abi.can_unwind()
        || abi.c_variadic()
        || !abi.hidden_arguments().is_empty()
        || abi.canon_abi() != SemanticCanonAbiV1::Rust
        || abi.extern_abi() != SemanticExternAbiV1::Rust
        || !matches!(call.unwind(), SemanticUnwindActionV1::Unreachable)
        || call.destination().is_none_or(|d| {
            !d.place().projections().is_empty() || d.place().ty() != abi.source_output_type()
        })
    {
        return refuse("closed Wave64 helper source signature unavailable");
    }
    let mut resolver = Resolver {
        owner,
        root,
        visits: 0,
    };
    if !resolver.is_f32(abi.source_output_type())
        || !resolver.is_f32(call.arguments()[2].ty())
        || abi.source_input_types()[2] != call.arguments()[2].ty()
    {
        return refuse("closed Wave64 helper scalar is not f32");
    }
    let mut meanings = [None; 2];
    let mut arguments = [None; 2];
    for index in 0..2 {
        let operand = &call.arguments()[index];
        if operand.ty() != abi.source_input_types()[index] {
            return refuse("actual helper source argument type changed");
        }
        let meaning = resolver.operand(
            caller,
            site(block),
            OperandRole::CallArgument(index as u32),
            operand,
            0,
            budget,
        )?;
        arguments[index] = Some(match (index, meaning.kind, meaning.referent) {
            (0, Kind::Snapshot { lane_field }, Some(snapshot_type)) => {
                if abi.source_argument_ownership()[index]
                    != SemanticSourceArgumentOwnershipV1::SharedBorrow
                {
                    return refuse("snapshot receiver is not original shared borrow");
                }
                ArgumentV1::Snapshot {
                    reference_type: meaning.ty,
                    snapshot_type,
                    lane_field,
                }
            }
            (0, Kind::LaneValue, None) if resolver.is_u32(meaning.ty) => {
                if abi.source_argument_ownership()[index]
                    != SemanticSourceArgumentOwnershipV1::ByValue
                {
                    return refuse("rank is not original by-value scalar");
                }
                ArgumentV1::Rank {
                    scalar_type: meaning.ty,
                }
            }
            (1, Kind::Context, Some(context_type)) => {
                if !resolver.zst(context_type) {
                    return refuse("issued context is not zero-state");
                }
                if abi.source_argument_ownership()[index]
                    != SemanticSourceArgumentOwnershipV1::SharedBorrow
                {
                    return refuse("context is not original shared borrow");
                }
                ArgumentV1::Context {
                    reference_type: meaning.ty,
                    context_type,
                }
            }
            _ => return refuse("helper reference role substitution"),
        });
        meanings[index] = Some(meaning);
    }
    if abi.source_argument_ownership()[2] != SemanticSourceArgumentOwnershipV1::ByValue {
        return refuse("f32 helper value is not by value");
    }
    Ok(CheckedReferenceCallV1 {
        owner,
        caller,
        block,
        helper: *helper,
        arguments: arguments.map(|v| v.expect("checked two original roles")),
        meanings: meanings.map(|v| v.expect("checked two original origins")),
    })
}

/// Every actual incoming source call agrees on one closed helper transport.
pub(super) struct CheckedReferenceHelperV1<'a> {
    representative: CheckedReferenceCallV1<'a>,
    incoming_count: usize,
}
impl<'a> CheckedReferenceHelperV1<'a> {
    pub(super) fn representative(&self) -> &CheckedReferenceCallV1<'a> {
        &self.representative
    }
    pub(super) fn incoming_count(&self) -> usize {
        self.incoming_count
    }
}
fn derive_helper<'a>(
    owner: &'a ProductionSemanticSsaOwnerV1,
    helper: SemanticFunctionIdV1,
    budget: &mut Budget<'_>,
) -> ResultV1<CheckedReferenceHelperV1<'a>> {
    let source = owner.source_semantic();
    if source.functions().len() > 32
        || source
            .functions()
            .get(helper.index() as usize)
            .is_none_or(|f| f.role() != SemanticFunctionRoleV1::InternalHelper)
    {
        return refuse("closed reference helper coordinate/role");
    }
    let mut representative: Option<CheckedReferenceCallV1<'a>> = None;
    let mut incoming_count = 0usize;
    for (fi, function) in source.functions().iter().enumerate() {
        for (bi, block) in function.blocks().iter().enumerate() {
            budget.charge_work(1)?;
            let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
                continue;
            };
            if !matches!(source.callables().get(call.callee().index() as usize),
                Some(SemanticCallableDeclV1::Defined { function }) if *function == helper)
            {
                continue;
            }
            // Unknown, retained, unreachable or foreign incoming calls are never skipped.
            let relation = derive(
                owner,
                SemanticFunctionIdV1::from_index(fi as u32),
                SemanticBlockIdV1::from_index(bi as u32),
                budget,
            )?;
            if let Some(previous) = &representative {
                if previous.arguments != relation.arguments
                    || !previous.same_source_origins(&relation)
                {
                    return refuse("helper incoming source origins or nominal roles disagree");
                }
            } else {
                representative = Some(relation);
            }
            incoming_count = incoming_count
                .checked_add(1)
                .ok_or(ReferenceErrorV1::Unavailable(
                    "helper incoming census overflow",
                ))?;
        }
    }
    Ok(CheckedReferenceHelperV1 {
        representative: representative.ok_or(ReferenceErrorV1::Unavailable(
            "helper has no original incoming call",
        ))?,
        incoming_count,
    })
}
/// The same callback must cover helper planning, emission and original call/formal replay.
pub(super) fn with_checked_reference_helper_v1<R: Copy + 'static>(
    owner: &ProductionSemanticSsaOwnerV1,
    helper: SemanticFunctionIdV1,
    budget: &mut Budget<'_>,
    inspect: impl FnOnce(&CheckedReferenceHelperV1<'_>, &mut Budget<'_>) -> ResultV1<R>,
) -> ResultV1<R> {
    if budget.failed_work().is_some() || budget.failed_storage().is_some() {
        return Err(Resource::Accounting.into());
    }
    let receipt = owner
        .occurrence_storage()
        .ok_or(ReferenceErrorV1::Unavailable(
            "captured source occurrences required",
        ))?;
    if budget.storage() < receipt.retained_storage() {
        return Err(Resource::Accounting.into());
    }
    let identity = budget.work_ledger_identity_v1();
    let slot = budget as *mut _ as usize;
    budget.reserve_storage(SCRATCH_V1)?;
    let floor = budget.storage();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let relation = derive_helper(owner, helper, budget)?;
        inspect(&relation, budget)
    }));
    if slot != budget as *mut _ as usize
        || identity != budget.work_ledger_identity_v1()
        || budget.storage() < floor
    {
        return Err(Resource::Accounting.into());
    }
    let denied = budget.failed_work().is_some() || budget.failed_storage().is_some();
    budget.release_storage(SCRATCH_V1)?;
    match outcome {
        Ok(Ok(_)) if denied => Err(Resource::Accounting.into()),
        Ok(result) => result,
        Err(payload) => {
            drop(payload);
            Err(ReferenceErrorV1::CallbackPanicked)
        }
    }
}

const SCRATCH_V1: usize = 64 * 4096;
/// Inspects the same immutable owner and original source occurrences. No view or
/// relation can escape; a consuming planner must run inside this callback.
pub(super) fn with_checked_reference_call_v1<R: Copy + 'static>(
    owner: &ProductionSemanticSsaOwnerV1,
    caller: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    budget: &mut Budget<'_>,
    inspect: impl FnOnce(&CheckedReferenceCallV1<'_>, &mut Budget<'_>) -> ResultV1<R>,
) -> ResultV1<R> {
    if budget.failed_work().is_some() || budget.failed_storage().is_some() {
        return Err(Resource::Accounting.into());
    }
    let receipt = owner
        .occurrence_storage()
        .ok_or(ReferenceErrorV1::Unavailable(
            "captured source occurrences required",
        ))?;
    if budget.storage() < receipt.retained_storage() {
        return Err(Resource::Accounting.into());
    }
    let identity = budget.work_ledger_identity_v1();
    let slot = budget as *mut _ as usize;
    budget.reserve_storage(SCRATCH_V1)?;
    let floor = budget.storage();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let relation = derive(owner, caller, block, budget)?;
        inspect(&relation, budget)
    }));
    if slot != budget as *mut _ as usize
        || identity != budget.work_ledger_identity_v1()
        || budget.storage() < floor
    {
        return Err(Resource::Accounting.into());
    }
    let denied = budget.failed_work().is_some() || budget.failed_storage().is_some();
    budget.release_storage(SCRATCH_V1)?;
    match outcome {
        Ok(Ok(_)) if denied => Err(Resource::Accounting.into()),
        Ok(result) => result,
        Err(payload) => {
            drop(payload);
            Err(ReferenceErrorV1::CallbackPanicked)
        }
    }
}

/// Exact root borrow/initializer/issuer coordinates, usable only while borrowing
/// the same source owner. This is arithmetic payload transport, never EXEC.
pub(super) struct CheckedReferenceBorrowV1<'a> {
    owner: &'a ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    statement: u32,
    source_rvalue: &'a SemanticRvalueV1,
    reference_type: SemanticTypeIdV1,
    snapshot_type: SemanticTypeIdV1,
    lane_field: u32,
    lane_producer: (SemanticFunctionIdV1, SsaValueV1),
    constructor: (SemanticBlockIdV1, u32),
}
impl CheckedReferenceBorrowV1<'_> {
    pub(super) fn owner(&self) -> &ProductionSemanticSsaOwnerV1 {
        self.owner
    }
    pub(super) fn function(&self) -> SemanticFunctionIdV1 {
        self.function
    }
    pub(super) fn block(&self) -> SemanticBlockIdV1 {
        self.block
    }
    pub(super) fn statement(&self) -> u32 {
        self.statement
    }
    pub(super) fn source_rvalue(&self) -> &SemanticRvalueV1 {
        self.source_rvalue
    }
    pub(super) fn reference_type(&self) -> SemanticTypeIdV1 {
        self.reference_type
    }
    pub(super) fn snapshot_type(&self) -> SemanticTypeIdV1 {
        self.snapshot_type
    }
    pub(super) fn lane_field(&self) -> u32 {
        self.lane_field
    }
    pub(super) fn lane_producer(&self) -> (SemanticFunctionIdV1, SsaValueV1) {
        self.lane_producer
    }
    pub(super) fn constructor(&self) -> (SemanticBlockIdV1, u32) {
        self.constructor
    }
    pub(super) fn initializer_rvalue(&self) -> &SemanticRvalueV1 {
        let statement = &self.owner.source_semantic().functions()[self.function.index() as usize]
            .blocks()[self.constructor.0.index() as usize]
            .statements()[self.constructor.1 as usize];
        let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
            unreachable!("immutable checked original initializer")
        };
        assignment.value()
    }
}
impl CheckedReferenceCallV1<'_> {
    pub(super) fn lane_producer(&self) -> (SemanticFunctionIdV1, SsaValueV1) {
        match self.meanings[0].origin {
            SourceOrigin::RootIssuedSsa { function, value } => (function, value),
            _ => unreachable!("checked arithmetic snapshot/rank retains actual root SSA issuer"),
        }
    }
    pub(super) fn context_producer(&self) -> Option<(SemanticFunctionIdV1, SsaValueV1)> {
        match self.meanings[1].origin {
            SourceOrigin::RootIssuedSsa { function, value } => Some((function, value)),
            _ => None,
        }
    }
    pub(super) fn retained_context_producer(
        &self,
    ) -> Option<(
        SemanticFunctionIdV1,
        SemanticBlockIdV1,
        SemanticLocalIdV1,
        SemanticTypeIdV1,
    )> {
        match self.meanings[1].origin {
            SourceOrigin::RootRetainedCall {
                function,
                block,
                local,
                context_type,
            } => Some((function, block, local, context_type)),
            _ => None,
        }
    }
}
fn derive_reference_borrow<'a>(
    owner: &'a ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    statement_index: u32,
    budget: &mut Budget<'_>,
) -> ResultV1<CheckedReferenceBorrowV1<'a>> {
    let source = owner.source_semantic();
    if source.roots() != [function] || source.functions().len() > 32 {
        return refuse("snapshot borrow accessor requires the sole original root");
    }
    let declaration = source
        .functions()
        .get(function.index() as usize)
        .ok_or(ReferenceErrorV1::Unavailable("source root coordinate"))?;
    if declaration.role() != SemanticFunctionRoleV1::KernelRoot
        || declaration.kernel_entry().is_none()
    {
        return refuse("snapshot borrow accessor requires actual kernel root");
    }
    let statement = declaration
        .blocks()
        .get(block.index() as usize)
        .and_then(|b| b.statements().get(statement_index as usize))
        .ok_or(ReferenceErrorV1::Unavailable(
            "original shared borrow coordinate",
        ))?;
    let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
        return refuse("snapshot borrow is not original source assignment");
    };
    let SemanticRvalueKindV1::Borrow {
        kind: SemanticBorrowKindV1::Shared,
        place,
    } = assignment.value().kind()
    else {
        return refuse("snapshot borrow site is not original Shared Borrow");
    };
    if !assignment.destination().projections().is_empty() || !place.projections().is_empty() {
        return refuse("snapshot borrow accessor requires exact root locals");
    }
    let borrow_site = Site::Statement {
        block: fe2o3_mir_model::SsaBlockIdV1::new(block.index()),
        statement: statement_index,
    };
    let mut original_definition = None;
    for event in rows(owner, function)?.events() {
        budget.charge_work(1)?;
        if event.site() == borrow_site
            && event.role() == EventRole::DestinationDefine
            && let Some(SsaResolvedEventV1::Define { variable, value }) = event.resolved()
            && variable.get() == assignment.destination().local().index()
        {
            if !event.is_reachable() || !event.is_promoted() {
                return refuse("source reference holder is unreachable or retained");
            }
            unique(&mut original_definition, value)?;
        }
    }
    if original_definition.is_none() {
        return refuse("source borrow has no exact original SSA definition");
    }
    let mut resolver = Resolver {
        owner,
        root: function,
        visits: 0,
    };
    // The retained-source proof is valid whether or not rustc also promoted this
    // particular aggregate. It never relabels addressed storage as an SSA value.
    let meaning = resolver.retained_snapshot(function, borrow_site, place, 0, budget)?;
    let Kind::Snapshot { lane_field } = meaning.kind else {
        return refuse("source borrow does not select original arithmetic snapshot");
    };
    let SourceOrigin::RootIssuedSsa {
        function: producer_function,
        value: producer_value,
    } = meaning.origin
    else {
        return refuse("root snapshot has no actual original lane SSA issuer");
    };
    if producer_function != function {
        return refuse("root snapshot does not have the current root's lane issuer");
    }
    resolver.shared(meaning, assignment.value().result_type())?;
    let mut constructor = None;
    for (bi, candidate_block) in declaration.blocks().iter().enumerate() {
        for (si, candidate) in candidate_block.statements().iter().enumerate() {
            budget.charge_work(1)?;
            if let SemanticStatementKindV1::Assign(candidate) = candidate.kind()
                && candidate.destination().local() == place.local()
            {
                // retained_snapshot independently established this exact unique,
                // dominating, whole aggregate initializer with no other writes.
                unique(
                    &mut constructor,
                    (SemanticBlockIdV1::from_index(bi as u32), si as u32),
                )?;
            }
        }
    }
    Ok(CheckedReferenceBorrowV1 {
        owner,
        function,
        block,
        statement: statement_index,
        source_rvalue: assignment.value(),
        reference_type: assignment.value().result_type(),
        snapshot_type: place.ty(),
        lane_field,
        lane_producer: (producer_function, producer_value),
        constructor: constructor.ok_or(ReferenceErrorV1::Unavailable(
            "original snapshot constructor absent",
        ))?,
    })
}
pub(super) fn with_checked_reference_borrow_v1<R: Copy + 'static>(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    statement: u32,
    budget: &mut Budget<'_>,
    inspect: impl FnOnce(&CheckedReferenceBorrowV1<'_>, &mut Budget<'_>) -> ResultV1<R>,
) -> ResultV1<R> {
    if budget.failed_work().is_some() || budget.failed_storage().is_some() {
        return Err(Resource::Accounting.into());
    }
    let receipt = owner
        .occurrence_storage()
        .ok_or(ReferenceErrorV1::Unavailable(
            "captured source occurrences required",
        ))?;
    if budget.storage() < receipt.retained_storage() {
        return Err(Resource::Accounting.into());
    }
    let identity = budget.work_ledger_identity_v1();
    let slot = budget as *mut _ as usize;
    budget.reserve_storage(SCRATCH_V1)?;
    let floor = budget.storage();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let relation = derive_reference_borrow(owner, function, block, statement, budget)?;
        inspect(&relation, budget)
    }));
    if slot != budget as *mut _ as usize
        || identity != budget.work_ledger_identity_v1()
        || budget.storage() < floor
    {
        return Err(Resource::Accounting.into());
    }
    let denied = budget.failed_work().is_some() || budget.failed_storage().is_some();
    budget.release_storage(SCRATCH_V1)?;
    match outcome {
        Ok(Ok(_)) if denied => Err(Resource::Accounting.into()),
        Ok(result) => result,
        Err(payload) => {
            drop(payload);
            Err(ReferenceErrorV1::CallbackPanicked)
        }
    }
}

/// One exact root SharedBorrow of retained context storage. The context is
/// authenticated by its original issuer call and lifetime, never by an SSA key.
pub(super) struct CheckedContextBorrowV1<'a> {
    owner: &'a ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    statement: u32,
    source_rvalue: &'a SemanticRvalueV1,
    reference_type: SemanticTypeIdV1,
    context_type: SemanticTypeIdV1,
    issuer_block: SemanticBlockIdV1,
    issuer_local: SemanticLocalIdV1,
}
impl CheckedContextBorrowV1<'_> {
    pub(super) fn owner(&self) -> &ProductionSemanticSsaOwnerV1 {
        self.owner
    }
    pub(super) fn function(&self) -> SemanticFunctionIdV1 {
        self.function
    }
    pub(super) fn block(&self) -> SemanticBlockIdV1 {
        self.block
    }
    pub(super) fn statement(&self) -> u32 {
        self.statement
    }
    pub(super) fn source_rvalue(&self) -> &SemanticRvalueV1 {
        self.source_rvalue
    }
    pub(super) fn reference_type(&self) -> SemanticTypeIdV1 {
        self.reference_type
    }
    pub(super) fn context_type(&self) -> SemanticTypeIdV1 {
        self.context_type
    }
    pub(super) fn issuer(&self) -> (SemanticFunctionIdV1, SemanticBlockIdV1, SemanticLocalIdV1) {
        (self.function, self.issuer_block, self.issuer_local)
    }
    pub(super) fn source_call(&self) -> &SemanticDirectCallV1 {
        call_at(self.owner, self.function, self.issuer_block)
            .expect("checked immutable root context issuer call")
    }
}
fn derive_context_borrow<'a>(
    owner: &'a ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    statement_index: u32,
    budget: &mut Budget<'_>,
) -> ResultV1<CheckedContextBorrowV1<'a>> {
    let source = owner.source_semantic();
    if source.roots() != [function] || source.functions().len() > 32 {
        return refuse("context borrow requires the sole original root");
    }
    let declaration = source
        .functions()
        .get(function.index() as usize)
        .ok_or(ReferenceErrorV1::Unavailable("source root coordinate"))?;
    if declaration.role() != SemanticFunctionRoleV1::KernelRoot
        || declaration.kernel_entry().is_none()
    {
        return refuse("context borrow requires actual kernel root");
    }
    let statement = declaration
        .blocks()
        .get(block.index() as usize)
        .and_then(|b| b.statements().get(statement_index as usize))
        .ok_or(ReferenceErrorV1::Unavailable(
            "original context borrow coordinate",
        ))?;
    let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
        return refuse("context borrow is not original source assignment");
    };
    let SemanticRvalueKindV1::Borrow {
        kind: SemanticBorrowKindV1::Shared,
        place,
    } = assignment.value().kind()
    else {
        return refuse("context borrow site is not original Shared Borrow");
    };
    if !assignment.destination().projections().is_empty() || !place.projections().is_empty() {
        return refuse("context borrow requires exact root locals");
    }
    let borrow_site = Site::Statement {
        block: fe2o3_mir_model::SsaBlockIdV1::new(block.index()),
        statement: statement_index,
    };
    let mut reference_definition = None;
    for event in rows(owner, function)?.events() {
        budget.charge_work(1)?;
        if event.site() == borrow_site
            && event.role() == EventRole::DestinationDefine
            && let Some(SsaResolvedEventV1::Define { variable, value }) = event.resolved()
            && variable.get() == assignment.destination().local().index()
        {
            if !event.is_reachable() || !event.is_promoted() {
                return refuse("context reference holder is unreachable or retained");
            }
            unique(&mut reference_definition, value)?;
        }
    }
    if reference_definition.is_none() {
        return refuse("context borrow has no exact original reference SSA definition");
    }
    let mut resolver = Resolver {
        owner,
        root: function,
        visits: 0,
    };
    let meaning = resolver.retained_context(function, borrow_site, place, 0, budget)?;
    resolver.shared(meaning, assignment.value().result_type())?;
    let SourceOrigin::RootRetainedCall {
        function: issuer_function,
        block: issuer_block,
        local: issuer_local,
        context_type,
    } = meaning.origin
    else {
        return refuse("context borrow requires original retained issuer call");
    };
    if meaning.kind != Kind::Context || issuer_function != function || context_type != place.ty() {
        return refuse("context borrow original issuer role/type changed");
    }
    Ok(CheckedContextBorrowV1 {
        owner,
        function,
        block,
        statement: statement_index,
        source_rvalue: assignment.value(),
        reference_type: assignment.value().result_type(),
        context_type,
        issuer_block,
        issuer_local,
    })
}
pub(super) fn with_checked_context_borrow_v1<R: Copy + 'static>(
    owner: &ProductionSemanticSsaOwnerV1,
    function: SemanticFunctionIdV1,
    block: SemanticBlockIdV1,
    statement: u32,
    budget: &mut Budget<'_>,
    inspect: impl FnOnce(&CheckedContextBorrowV1<'_>, &mut Budget<'_>) -> ResultV1<R>,
) -> ResultV1<R> {
    if budget.failed_work().is_some() || budget.failed_storage().is_some() {
        return Err(Resource::Accounting.into());
    }
    let receipt = owner
        .occurrence_storage()
        .ok_or(ReferenceErrorV1::Unavailable(
            "captured source occurrences required",
        ))?;
    if budget.storage() < receipt.retained_storage() {
        return Err(Resource::Accounting.into());
    }
    let identity = budget.work_ledger_identity_v1();
    let slot = budget as *mut _ as usize;
    budget.reserve_storage(SCRATCH_V1)?;
    let floor = budget.storage();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let relation = derive_context_borrow(owner, function, block, statement, budget)?;
        inspect(&relation, budget)
    }));
    if slot != budget as *mut _ as usize
        || identity != budget.work_ledger_identity_v1()
        || budget.storage() < floor
    {
        return Err(Resource::Accounting.into());
    }
    let denied = budget.failed_work().is_some() || budget.failed_storage().is_some();
    budget.release_storage(SCRATCH_V1)?;
    match outcome {
        Ok(Ok(_)) if denied => Err(Resource::Accounting.into()),
        Ok(result) => result,
        Err(payload) => {
            drop(payload);
            Err(ReferenceErrorV1::CallbackPanicked)
        }
    }
}

#[cfg(test)]
#[path = "production_wave64_reference_call_v1/tests.rs"]
mod tests;
