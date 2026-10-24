//! Private retained first argument writer. No ordinary route or recipe readiness.
//! HashMap lookup-only semantics become charged source-ordered unique rows.
use super::*;
use std::mem::size_of;
#[path = "retained_initial_strided_read_frame_v1.rs"]
mod frame;
#[path = "whole_root_initial_strided_read_bridge_v1.rs"]
mod whole;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReadPhase {
    Fresh,
    Terminal,
    Complete,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReadSourceKey {
    function: usize,
    types: (usize, usize),
    effects: (usize, usize),
    origins: (usize, usize),
}
impl ReadSourceKey {
    fn new(
        types: &[SemanticTypeDeclV1],
        function: &SemanticFunctionDeclV1,
        effects: &[Option<ProjectedReadViewAccessV1>],
        origins: &[Option<u32>],
    ) -> Self {
        Self {
            function: function as *const _ as usize,
            types: (types.as_ptr() as usize, types.len()),
            effects: (effects.as_ptr() as usize, effects.len()),
            origins: (origins.as_ptr() as usize, origins.len()),
        }
    }
}
struct ReadViewRow {
    source: ProjectedReadViewV1,
    view: ProjectedViewV1,
}
pub(super) struct RetainedInitialStridedReadV1 {
    phase: ReadPhase,
    source: Option<ReadSourceKey>,
    entry: Option<Snapshot>,
    held: Option<Snapshot>,
    failure: Option<Backend>,
    views: Vec<ReadViewRow>,
    pending_view: Option<ReadViewRow>,
    projected: Vec<Option<GuardedRankedAccessV1>>,
    lookup_visits: usize,
    completed_blocks: usize,
    final_next_value: u32,
    final_next_argument: usize,
    final_operations: usize,
}
impl RetainedInitialStridedReadV1 {
    pub(super) fn new() -> Self {
        Self {
            phase: ReadPhase::Fresh,
            source: None,
            entry: None,
            held: None,
            failure: None,
            views: Vec::new(),
            pending_view: None,
            projected: Vec::new(),
            lookup_visits: 0,
            completed_blocks: 0,
            final_next_value: 0,
            final_next_argument: 0,
            final_operations: 0,
        }
    }
    fn saved(&self) -> Backend {
        match &self.failure {
            Some(Backend::CanonicalAssertions(CanonicalAssertionErrorV1::Resource(error))) => {
                resource(*error)
            }
            Some(Backend::Unsupported(detail)) => Backend::Unsupported(*detail),
            Some(Backend::Incomplete(detail)) => Backend::Incomplete(*detail),
            _ => accounting(),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_into(
        &mut self,
        types: &[SemanticTypeDeclV1],
        function: &SemanticFunctionDeclV1,
        effects: &[Option<ProjectedReadViewAccessV1>],
        origins: &[Option<u32>],
        arguments: &mut RetainedBeforeArgumentWritersV1,
        prefix: &mut RootEntryPrefixV1,
        resources: &mut Prep<'_, '_>,
    ) -> BResult<()> {
        let fresh = self.phase == ReadPhase::Fresh;
        self.phase = ReadPhase::Terminal;
        if !fresh {
            return Err(self.saved());
        }
        self.source = Some(ReadSourceKey::new(types, function, effects, origins));
        self.entry = resources.retained_custody_snapshot_v1();
        let result = self.prepare_original(
            types, function, effects, origins, arguments, prefix, resources,
        );
        match result {
            Ok(()) => {
                self.held = resources.retained_custody_snapshot_v1();
                self.phase = ReadPhase::Complete;
                Ok(())
            }
            Err(error) => {
                self.failure = Some(error);
                Err(self.saved())
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn prepare_original(
        &mut self,
        types: &[SemanticTypeDeclV1],
        function: &SemanticFunctionDeclV1,
        effects: &[Option<ProjectedReadViewAccessV1>],
        origins: &[Option<u32>],
        arguments: &mut RetainedBeforeArgumentWritersV1,
        prefix: &mut RootEntryPrefixV1,
        resources: &mut Prep<'_, '_>,
    ) -> BResult<()> {
        self.check_custody(resources)?;
        resources.work(32)?;
        resources.reserve_storage(frame::bytes()?)?;
        if effects.len() != function.blocks().len() {
            return Err(Backend::Unsupported(
                "strided read effects do not correspond one-to-one with semantic MIR blocks",
            ));
        }
        if effects.len() > 32
            || function.locals().len() > 4096
            || origins.len() != function.locals().len()
        {
            return Err(Backend::Incomplete(
                "retained initial read exceeds closed source profile",
            ));
        }
        arguments.before_writers(function.locals().len(), resources)?;
        resources.reserve(&mut self.projected, effects.len())?;
        for (block, effect) in function.blocks().iter().zip(effects.iter().copied()) {
            resources.work(1)?;
            let Some(effect) = effect else {
                // Capacity was admitted for the complete row array above.
                self.projected.push(None);
                self.completed_blocks += 1;
                continue;
            };
            if effect.view.allocation.writable {
                return Err(Backend::Unsupported(
                    "a checked shared read view rooted in writable allocation authority",
                ));
            }
            let existing = self.find(effect.view.root, resources)?;
            let (view, rows, columns) = if let Some(index) = existing {
                let row = &self.views[index];
                resources.work(1)?;
                if row.source != effect.view {
                    return Err(Backend::Unsupported(
                        "one checked read-view origin changed its element, extent, or allocation contract",
                    ));
                }
                let [rows, columns] = row.view.dynamic_extents.as_slice() else {
                    return Err(Backend::Unsupported(
                        "a checked read view without exactly two dynamic extents",
                    ));
                };
                (row.view.result, *rows, *columns)
            } else {
                let rows = read_value(effect.view.rows, origins, arguments, prefix, resources)?;
                let columns =
                    read_value(effect.view.columns, origins, arguments, prefix, resources)?;
                // Preserve SSA assignment and type validation order from the donor.
                resources.work(1)?;
                reserve_operation_paid(&mut prefix.entry_operations, resources)?;
                let result = next_value_id(&mut prefix.next_value)?;
                resources.work(1)?;
                let width = type_width(types, effect.view.element)?;
                self.pending_view = Some(ReadViewRow {
                    source: effect.view,
                    view: ProjectedViewV1 {
                        result,
                        element_width: width,
                        writable: false,
                        shape: Vec::new(),
                        dynamic_extents: Vec::new(),
                        memory_space: MemorySpaceAttr::Global,
                        allocation_origin: effect.view.allocation.allocation_origin,
                        noalias_class: effect.view.allocation.noalias_class,
                    },
                });
                let pending = self.pending_view.as_mut().ok_or_else(accounting)?;
                resources.push(&mut pending.view.shape, DYNAMIC_EXTENT)?;
                resources.push(&mut pending.view.shape, DYNAMIC_EXTENT)?;
                resources.push(&mut pending.view.dynamic_extents, rows)?;
                resources.push(&mut pending.view.dynamic_extents, columns)?;
                // Outer operation capacity and its append work were already paid.
                // Attach both empty clone destinations before either can allocate.
                let operation = prefix.entry_operations.len();
                prefix
                    .entry_operations
                    .push(ProductionRankedOperationV1::ViewInSpace {
                        result,
                        element_width: width,
                        writable: false,
                        shape: Vec::new(),
                        dynamic_extents: Vec::new(),
                        memory_space: MemorySpaceAttr::Global,
                        allocation_origin: effect.view.allocation.allocation_origin,
                        noalias_class: effect.view.allocation.noalias_class,
                    });
                let ProductionRankedOperationV1::ViewInSpace {
                    shape,
                    dynamic_extents,
                    ..
                } = &mut prefix.entry_operations[operation]
                else {
                    return Err(accounting());
                };
                // No owned payload is placed in a fallible push call.
                for value in pending.view.shape.iter().copied() {
                    resources.push(shape, value)?;
                }
                for value in pending.view.dynamic_extents.iter().copied() {
                    resources.push(dynamic_extents, value)?;
                }
                // Original map insertion occurs only after the operation clone.
                // Pay every fallible action before the retained payload is moved.
                resources.work(1)?;
                resources.reserve(&mut self.views, 1)?;
                let Some(row) = self.pending_view.take() else {
                    return Err(accounting());
                };
                self.views.push(row);
                (result, rows, columns)
            };
            let row = read_value(effect.row, origins, arguments, prefix, resources)?;
            let column = read_value(effect.column, origins, arguments, prefix, resources)?;
            resources.work(1)?;
            let index = self.projected.len();
            self.projected.push(Some(GuardedRankedAccessV1 {
                failure: GuardedAccessFailureV1::ReturnFallback,
                view,
                indices: Vec::new(),
                checked_success: None,
                comparisons: Vec::new(),
                access: AccessKindAttr::Read,
                memory_space: MemorySpaceAttr::Global,
                source: block.terminator().source(),
                output_extent: None,
                semantic_site: None,
            }));
            let access = self.projected[index].as_mut().ok_or_else(accounting)?;
            resources.push(&mut access.indices, row)?;
            resources.push(&mut access.indices, column)?;
            resources.push(&mut access.comparisons, (row, rows))?;
            resources.push(&mut access.comparisons, (column, columns))?;
            self.completed_blocks += 1;
        }
        self.check_custody(resources)?;
        if self.pending_view.is_some() || self.completed_blocks != effects.len() {
            return Err(accounting());
        }
        self.final_next_value = prefix.next_value;
        self.final_next_argument = arguments.next_runtime_argument;
        self.final_operations = prefix.entry_operations.len();
        Ok(())
    }
    fn find(&mut self, root: u64, resources: &mut Prep<'_, '_>) -> BResult<Option<usize>> {
        for (index, row) in self.views.iter().enumerate() {
            resources.work(1)?;
            self.lookup_visits = self.lookup_visits.checked_add(1).ok_or_else(arithmetic)?;
            if row.source.root == root {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
    fn check_custody(&self, resources: &Prep<'_, '_>) -> BResult<()> {
        let entry = self.entry.ok_or_else(accounting)?;
        let now = resources
            .retained_custody_snapshot_v1()
            .ok_or_else(accounting)?;
        let growth = now.owned.checked_sub(entry.owned).ok_or_else(accounting)?;
        if now.budget_slot != entry.budget_slot
            || now.work_ledger != entry.work_ledger
            || now.owned_slot != entry.owned_slot
            || now.storage != entry.storage.checked_add(growth).ok_or_else(arithmetic)?
            || now.work < entry.work
            || now.peak < entry.peak
            || entry.denied_work
            || entry.denied_storage
            || entry.owned > entry.storage
            || now.denied_work
            || now.denied_storage
        {
            return Err(accounting());
        }
        if let Some(held) = self.held {
            if now.owned < held.owned
                || now.storage < held.storage
                || now.work < held.work
                || now.peak < held.peak
            {
                return Err(accounting());
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn completed_for(
        &self,
        types: &[SemanticTypeDeclV1],
        function: &SemanticFunctionDeclV1,
        effects: &[Option<ProjectedReadViewAccessV1>],
        origins: &[Option<u32>],
        arguments: &RetainedBeforeArgumentWritersV1,
        prefix: &RootEntryPrefixV1,
        resources: &Prep<'_, '_>,
    ) -> BResult<()> {
        if self.phase != ReadPhase::Complete
            || self.failure.is_some()
            || self.source != Some(ReadSourceKey::new(types, function, effects, origins))
            || self.pending_view.is_some()
            || self.completed_blocks != effects.len()
            || self.projected.len() != effects.len()
            || self.final_next_value != prefix.next_value
            || self.final_next_argument != arguments.next_runtime_argument
            || self.final_operations != prefix.entry_operations.len()
        {
            return Err(self.saved());
        }
        self.check_custody(resources)
    }
}
fn read_value(
    value: ProjectedReadValueV1,
    origins: &[Option<u32>],
    arguments: &mut RetainedBeforeArgumentWritersV1,
    prefix: &mut RootEntryPrefixV1,
    resources: &mut Prep<'_, '_>,
) -> BResult<ProductionRankedValueV1> {
    resources.work(1)?;
    match value {
        ProjectedReadValueV1::Constant(value) => {
            resources.work(1)?;
            reserve_operation_paid(&mut prefix.entry_operations, resources)?;
            let result = next_value_id(&mut prefix.next_value)?;
            prefix
                .entry_operations
                .push(ProductionRankedOperationV1::IndexConstant { result, value });
            Ok(ProductionRankedValueV1::Local(result))
        }
        ProjectedReadValueV1::Local(local) => {
            let origin = origins
                .get(local.index() as usize)
                .copied()
                .flatten()
                .unwrap_or(local.index()) as usize;
            let slot =
                arguments
                    .runtime_index_arguments
                    .get_mut(origin)
                    .ok_or(Backend::Unsupported(
                        "a strided read index origin outside the semantic local table",
                    ))?;
            let argument = match *slot {
                Some(argument) => argument,
                None => {
                    let argument =
                        u32::try_from(arguments.next_runtime_argument).map_err(|_| {
                            Backend::Unsupported("too many strided read ranked arguments")
                        })?;
                    arguments.next_runtime_argument =
                        arguments.next_runtime_argument.checked_add(1).ok_or(
                            Backend::Unsupported("strided read ranked argument count overflow"),
                        )?;
                    *slot = Some(argument);
                    argument
                }
            };
            Ok(ProductionRankedValueV1::Argument(argument))
        }
    }
}
fn reserve_operation_paid(
    operations: &mut Vec<ProductionRankedOperationV1>,
    resources: &mut Prep<'_, '_>,
) -> BResult<()> {
    if operations.len() == MAX_PROJECTED_OPERATIONS_V1 {
        return Err(Backend::Unsupported(
            "a semantic intrinsic projection exceeding the ranked operation limit",
        ));
    }
    resources.reserve(operations, 1)
}
#[cfg(test)]
#[path = "retained_initial_strided_read_v1_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "retained_initial_empty_read_genuine_v1_tests.rs"]
pub(super) mod genuine_empty;

// Independent retained legacy-donor controls; no actual-source entry is wired here.
#[cfg(test)]
#[path = "retained_strided_read_original_v1_tests.rs"]
pub(super) mod retained_original;

#[cfg(test)]
#[path = "retained_initial_nonempty_read_genuine_v1_tests.rs"]
pub(super) mod genuine_nonempty;
