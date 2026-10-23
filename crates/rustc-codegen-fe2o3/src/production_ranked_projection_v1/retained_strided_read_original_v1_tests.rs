//! Test-only retained transcription of the legacy strided-read donor.
//! This owns the donor HashMap and every partial nested payload outside the
//! checked-source callback. It does not invoke or read the retained candidate.
//! Logical map slots are metered, not allocator internals, hash probes, or RSS.
use super::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OriginalPhase {
    Fresh,
    Terminal,
    Complete,
}

type OriginalView = (ProjectedReadViewV1, ProjectedViewV1);

pub(in super::super) struct RetainedOriginalReadsV1 {
    phase: OriginalPhase,
    source: Option<ReadSourceKey>,
    entry: Option<Snapshot>,
    held: Option<Snapshot>,
    failure: Option<Backend>,
    pub(in super::super) views: HashMap<u64, OriginalView>,
    pending_view: Option<OriginalView>,
    displaced_view: Option<OriginalView>,
    pub(in super::super) projected: Vec<Option<GuardedRankedAccessV1>>,
    pub(in super::super) operations: Vec<ProductionRankedOperationV1>,
    pub(in super::super) arguments: Vec<Option<u32>>,
    pub(in super::super) next_value: u32,
    pub(in super::super) next_argument: usize,
    completed_blocks: usize,
    map_slots_debited: usize,
}

impl RetainedOriginalReadsV1 {
    pub(in super::super) fn new() -> Self {
        Self {
            phase: OriginalPhase::Fresh,
            source: None,
            entry: None,
            held: None,
            failure: None,
            views: HashMap::new(),
            pending_view: None,
            displaced_view: None,
            projected: Vec::new(),
            operations: Vec::new(),
            arguments: Vec::new(),
            next_value: 0,
            next_argument: 1,
            completed_blocks: 0,
            map_slots_debited: 0,
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
    pub(in super::super) fn prepare_into(
        &mut self,
        types: &[SemanticTypeDeclV1],
        function: &SemanticFunctionDeclV1,
        effects: &[Option<ProjectedReadViewAccessV1>],
        origins: &[Option<u32>],
        initial_layout: Option<&ProductionRankedOperationV1>,
        resources: &mut Prep<'_, '_>,
    ) -> BResult<()> {
        let fresh = self.phase == OriginalPhase::Fresh;
        self.phase = OriginalPhase::Terminal;
        if !fresh {
            return Err(self.saved());
        }
        self.source = Some(ReadSourceKey::new(types, function, effects, origins));
        self.entry = resources.retained_custody_snapshot_v1();
        let result =
            self.prepare_original(types, function, effects, origins, initial_layout, resources);
        match result {
            Ok(()) => {
                self.held = resources.retained_custody_snapshot_v1();
                self.phase = OriginalPhase::Complete;
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
        initial_layout: Option<&ProductionRankedOperationV1>,
        resources: &mut Prep<'_, '_>,
    ) -> BResult<()> {
        self.check_custody(resources)?;
        resources.work(32)?;
        resources.reserve_storage(frame()?)?;
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
                "retained original reads exceed closed source profile",
            ));
        }
        if let Some(layout) = initial_layout {
            resources.work(1)?;
            // Only this flat, allocation-free operation may be copied here.
            if !matches!(layout, ProductionRankedOperationV1::ExecutionLayout { .. }) {
                return Err(Backend::Incomplete(
                    "retained original reads require an execution-only prefix",
                ));
            }
            resources.push(&mut self.operations, layout.clone())?;
        }
        resources.reserve(&mut self.arguments, function.locals().len())?;
        for _ in function.locals() {
            resources.push(&mut self.arguments, None)?;
        }
        resources.reserve(&mut self.projected, effects.len())?;

        // Keep the legacy block order and HashMap lookup semantics. Metering
        // changes allocation custody, not source evaluation or SSA order.
        for (block, effect) in function.blocks().iter().zip(effects.iter().copied()) {
            resources.work(1)?;
            let Some(effect) = effect else {
                self.projected.push(None);
                self.completed_blocks += 1;
                continue;
            };
            if effect.view.allocation.writable {
                return Err(Backend::Unsupported(
                    "a checked shared read view rooted in writable allocation authority",
                ));
            }
            resources.work(1)?;
            let (view, rows, columns) = if let Some((source, view)) =
                self.views.get(&effect.view.root)
            {
                resources.work(1)?;
                if *source != effect.view {
                    return Err(Backend::Unsupported(
                        "one checked read-view origin changed its element, extent, or allocation contract",
                    ));
                }
                let [rows, columns] = view.dynamic_extents.as_slice() else {
                    return Err(Backend::Unsupported(
                        "a checked read view without exactly two dynamic extents",
                    ));
                };
                (view.result, *rows, *columns)
            } else {
                let rows = self.read_value(effect.view.rows, origins, resources)?;
                let columns = self.read_value(effect.view.columns, origins, resources)?;
                self.reserve_operation(resources)?;
                let result = next_value_id(&mut self.next_value)?;
                resources.work(1)?;
                let width = type_width(types, effect.view.element)?;
                self.pending_view = Some((
                    effect.view,
                    ProjectedViewV1 {
                        result,
                        element_width: width,
                        writable: false,
                        shape: Vec::new(),
                        dynamic_extents: Vec::new(),
                        memory_space: MemorySpaceAttr::Global,
                        allocation_origin: effect.view.allocation.allocation_origin,
                        noalias_class: effect.view.allocation.noalias_class,
                    },
                ));
                let (_, pending) = self.pending_view.as_mut().ok_or_else(accounting)?;
                resources.push(&mut pending.shape, DYNAMIC_EXTENT)?;
                resources.push(&mut pending.shape, DYNAMIC_EXTENT)?;
                resources.push(&mut pending.dynamic_extents, rows)?;
                resources.push(&mut pending.dynamic_extents, columns)?;

                // Outer operation capacity was admitted before result/type
                // validation. Attach the empty clone before either Vec fills.
                let operation = self.operations.len();
                self.operations
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
                } = &mut self.operations[operation]
                else {
                    return Err(accounting());
                };
                for value in pending.shape.iter().copied() {
                    resources.push(shape, value)?;
                }
                for value in pending.dynamic_extents.iter().copied() {
                    resources.push(dynamic_extents, value)?;
                }

                // Both map-reservation and any observed capacity excess are
                // paid while the source payload is still pending on this owner.
                self.reserve_map_slot(resources)?;
                resources.work(1)?;
                let row = self.pending_view.take().ok_or_else(accounting)?;
                self.displaced_view = self.views.insert(effect.view.root, row);
                if self.displaced_view.is_some() {
                    return Err(accounting());
                }
                (result, rows, columns)
            };
            let row = self.read_value(effect.row, origins, resources)?;
            let column = self.read_value(effect.column, origins, resources)?;
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
        if self.pending_view.is_some()
            || self.displaced_view.is_some()
            || self.completed_blocks != effects.len()
            || self.map_slots_debited != self.views.capacity()
        {
            return Err(accounting());
        }
        Ok(())
    }

    fn reserve_map_slot(&mut self, resources: &mut Prep<'_, '_>) -> BResult<()> {
        resources.work(1)?;
        let before = self.views.capacity();
        if before != self.map_slots_debited {
            return Err(accounting());
        }
        let required = self.views.len().checked_add(1).ok_or_else(arithmetic)?;
        let requested = required.saturating_sub(before);
        let bytes = requested
            .checked_mul(size_of::<(u64, OriginalView)>())
            .ok_or_else(arithmetic)?;
        resources.reserve_storage(bytes)?;
        self.map_slots_debited = before.checked_add(requested).ok_or_else(arithmetic)?;
        self.views
            .try_reserve(1)
            .map_err(|_| resource(Resource::Allocation))?;
        // The map is already an outer-owned field if this excess debit fails.
        let excess = self
            .views
            .capacity()
            .checked_sub(self.map_slots_debited)
            .ok_or_else(accounting)?;
        resources.reserve_storage(
            excess
                .checked_mul(size_of::<(u64, OriginalView)>())
                .ok_or_else(arithmetic)?,
        )?;
        self.map_slots_debited = self.views.capacity();
        Ok(())
    }

    fn reserve_operation(&mut self, resources: &mut Prep<'_, '_>) -> BResult<()> {
        resources.work(1)?;
        if self.operations.len() == MAX_PROJECTED_OPERATIONS_V1 {
            return Err(Backend::Unsupported(
                "a semantic intrinsic projection exceeding the ranked operation limit",
            ));
        }
        resources.reserve(&mut self.operations, 1)
    }

    fn read_value(
        &mut self,
        value: ProjectedReadValueV1,
        origins: &[Option<u32>],
        resources: &mut Prep<'_, '_>,
    ) -> BResult<ProductionRankedValueV1> {
        resources.work(1)?;
        match value {
            ProjectedReadValueV1::Constant(value) => {
                self.reserve_operation(resources)?;
                let result = next_value_id(&mut self.next_value)?;
                self.operations
                    .push(ProductionRankedOperationV1::IndexConstant { result, value });
                Ok(ProductionRankedValueV1::Local(result))
            }
            ProjectedReadValueV1::Local(local) => {
                let local_index = local.index() as usize;
                let origin = origins
                    .get(local_index)
                    .copied()
                    .flatten()
                    .unwrap_or(local.index()) as usize;
                let slot = self.arguments.get_mut(origin).ok_or(Backend::Unsupported(
                    "a strided read index origin outside the semantic local table",
                ))?;
                let argument = match *slot {
                    Some(argument) => argument,
                    None => {
                        let argument = u32::try_from(self.next_argument).map_err(|_| {
                            Backend::Unsupported("too many strided read ranked arguments")
                        })?;
                        self.next_argument =
                            self.next_argument
                                .checked_add(1)
                                .ok_or(Backend::Unsupported(
                                    "strided read ranked argument count overflow",
                                ))?;
                        *slot = Some(argument);
                        argument
                    }
                };
                Ok(ProductionRankedValueV1::Argument(argument))
            }
        }
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

    pub(in super::super) fn completed_for(
        &self,
        types: &[SemanticTypeDeclV1],
        function: &SemanticFunctionDeclV1,
        effects: &[Option<ProjectedReadViewAccessV1>],
        origins: &[Option<u32>],
        resources: &Prep<'_, '_>,
    ) -> BResult<()> {
        if self.phase != OriginalPhase::Complete
            || self.failure.is_some()
            || self.source != Some(ReadSourceKey::new(types, function, effects, origins))
            || self.pending_view.is_some()
            || self.displaced_view.is_some()
            || self.projected.len() != effects.len()
            || self.arguments.len() != function.locals().len()
            || self.completed_blocks != effects.len()
            || self.map_slots_debited != self.views.capacity()
        {
            return Err(self.saved());
        }
        self.check_custody(resources)
    }
}

#[path = "retained_strided_read_original_frame_v1_tests.rs"]
mod frame_policy;
pub(in super::super) fn frame() -> BResult<usize> {
    frame_policy::bytes()
}
#[path = "retained_strided_read_original_controls_v1_tests.rs"]
mod controls;
