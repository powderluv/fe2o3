//! Conservative analyses over [`fe2o3_kernel_ir`].
//!
//! This crate reports analysis facts and rejected obligations. It does not
//! grant `Checked`, `Verified`, safe-launch, or any other assurance authority.

#![deny(unsafe_code)]

#[cfg(feature = "authenticated-machine-effect")]
mod authenticated_machine_effect;
mod canonical_kir_call_effects_v1;
mod canonical_kir_contract_catalog_v1;
mod canonical_kir_coordinate_preservation_v1;
mod canonical_kir_cross_block_forwarding_v1;
mod canonical_kir_induction_refinement_v1;
mod canonical_kir_inventory_v1;
mod canonical_kir_licm_v1;
mod canonical_kir_load_forwarding_v1;
mod canonical_kir_loop_preheaders_v1;
mod canonical_kir_loop_unroll_v1;
mod canonical_kir_loops_v1;
mod canonical_kir_memory_ssa_v1;
mod canonical_kir_must_alias_v1;
mod canonical_kir_physical_occurrences_v1;
mod canonical_kir_private_cell_census_v1;
mod canonical_kir_private_cell_pair_resources_v1;
mod canonical_kir_private_cell_promotion_v1;
mod canonical_kir_private_memory_v1;
mod canonical_kir_redundant_store_v1;
mod canonical_kir_same_cfg_payload_v1;
mod canonical_kir_sparse_scalar_v1;
mod canonical_kir_sparse_v1;
mod canonical_kir_store_forwarding_v1;
mod canonical_kir_transition_v1;
mod canonical_tile_convergence_v160;
mod control_flow;
mod formal_path_conflicts_v1;
#[cfg(feature = "authenticated-machine-effect")]
mod gfx942_atomic_collective_structure;
#[cfg(feature = "authenticated-machine-effect")]
mod gfx942_exec_control;
#[cfg(feature = "authenticated-machine-effect")]
mod gfx942_fill_analysis_v1;
#[cfg(feature = "gfx942-fill-model")]
mod gfx942_fill_wave_v1;
#[cfg(feature = "authenticated-machine-effect")]
mod gfx942_machine_dataflow;
#[cfg(feature = "authenticated-machine-effect")]
mod gfx942_machine_register;
#[cfg(feature = "authenticated-machine-effect")]
mod gfx942_scalar_f32_recurrence;
mod kernel_check_model;
#[cfg(feature = "authenticated-machine-effect")]
mod physical_machine_analysis;
#[cfg(feature = "authenticated-machine-effect")]
mod physical_machine_effect;
#[cfg(feature = "authenticated-machine-effect")]
mod physical_machine_trace;
mod pliron_presburger;
#[cfg(feature = "authenticated-machine-effect")]
mod process_execution;
mod ssa;
mod uniformity;

#[cfg(feature = "authenticated-machine-effect")]
pub use authenticated_machine_effect::*;
pub use canonical_kir_call_effects_v1::*;
pub use canonical_kir_contract_catalog_v1::*;
pub use canonical_kir_coordinate_preservation_v1::*;
pub use canonical_kir_cross_block_forwarding_v1::*;
pub use canonical_kir_induction_refinement_v1::*;
pub use canonical_kir_inventory_v1::*;
pub use canonical_kir_licm_v1::*;
pub use canonical_kir_load_forwarding_v1::*;
pub use canonical_kir_loop_preheaders_v1::*;
pub use canonical_kir_loop_unroll_v1::*;
pub use canonical_kir_loops_v1::*;
pub use canonical_kir_memory_ssa_v1::*;
pub use canonical_kir_must_alias_v1::*;
pub use canonical_kir_physical_occurrences_v1::*;
pub use canonical_kir_private_cell_census_v1::*;
pub use canonical_kir_private_cell_promotion_v1::*;
pub use canonical_kir_private_memory_v1::*;
pub use canonical_kir_redundant_store_v1::*;
pub use canonical_kir_sparse_v1::*;
pub use canonical_kir_store_forwarding_v1::*;
pub use canonical_kir_transition_v1::*;
pub use canonical_tile_convergence_v160::*;
pub use control_flow::{
    ControlFlowAnalysis, ControlFlowDiagnostic, ControlFlowDiagnosticV2, ControlFlowEdge,
    ControlFlowErrors, ControlFlowResource, ControlFlowResourceUsage, MAX_CONTROL_FLOW_BLOCKS,
    MAX_CONTROL_FLOW_DOMINANCE_FRONTIER_ENTRIES, MAX_CONTROL_FLOW_EDGES,
    MAX_CONTROL_FLOW_IDF_ENTRIES, MAX_CONTROL_FLOW_LOOP_BODY_MEMBERSHIPS,
    MAX_CONTROL_FLOW_NATURAL_LOOPS, MAX_CONTROL_FLOW_STORAGE_ITEMS, MAX_CONTROL_FLOW_WORK_UNITS,
    MAX_SSA_PLACEMENT_OUTPUT_ITEMS, analyze_control_flow,
};
pub use formal_path_conflicts_v1::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use gfx942_atomic_collective_structure::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use gfx942_exec_control::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use gfx942_fill_analysis_v1::*;
#[cfg(feature = "gfx942-fill-model")]
pub use gfx942_fill_wave_v1::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use gfx942_machine_dataflow::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use gfx942_machine_register::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use gfx942_scalar_f32_recurrence::*;
pub use kernel_check_model::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use physical_machine_analysis::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use physical_machine_effect::*;
#[cfg(feature = "authenticated-machine-effect")]
pub use physical_machine_trace::*;
pub use pliron_presburger::*;
pub use ssa::{
    SsaPlacement, SsaPlacementDiagnostic, SsaPlacementErrors, SsaVariable, SsaVariablePlacement,
    place_pruned_ssa_parameters,
};
pub use uniformity::{
    UniformityPhysicalLaunchErrorV2, UniformityPhysicalLaunchV2, analyze_function,
    analyze_kernel_entry, analyze_module_helper_v1,
};

use fe2o3_kernel_ir::{BlockId, FunctionId, SynchronizationScope, ValueId};
use std::collections::BTreeMap;

/// How broadly a value is guaranteed to agree across a launch hierarchy.
///
/// The ordering is the variation lattice from `gpu-safety-contract-v1`:
/// grid-uniform values are least varying and invocation-varying values are
/// most varying. [`Variation::join`] computes the least upper bound.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Variation {
    GridUniform,
    WorkgroupUniform,
    SubgroupUniform,
    Varying,
}

impl Variation {
    /// Returns the least upper bound of two variation classifications.
    pub const fn join(self, other: Self) -> Self {
        if self as u8 >= other as u8 {
            self
        } else {
            other
        }
    }

    /// Returns whether this variation is uniform enough for all participants
    /// at `scope` to make the same control-flow decision.
    pub const fn is_uniform_for(self, scope: SynchronizationScope) -> bool {
        let maximum = match scope {
            SynchronizationScope::Invocation => Self::Varying,
            SynchronizationScope::Subgroup => Self::SubgroupUniform,
            SynchronizationScope::Workgroup => Self::WorkgroupUniform,
            SynchronizationScope::Device | SynchronizationScope::System => Self::GridUniform,
        };
        (self as u8) <= (maximum as u8)
    }
}

/// A deterministic result from one function analysis.
///
/// Missing facts must not be interpreted as uniform. Use [`Self::value`] and
/// [`Self::block_control`] to obtain conservative `Varying` defaults.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisReport {
    pub(crate) function: FunctionId,
    pub(crate) values: BTreeMap<ValueId, Variation>,
    pub(crate) block_controls: BTreeMap<BlockId, Variation>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

impl AnalysisReport {
    /// The analyzed function identity.
    pub fn function(&self) -> &FunctionId {
        &self.function
    }

    /// The value's variation, defaulting to `Varying` when no fact exists.
    pub fn value(&self, value: ValueId) -> Variation {
        self.values
            .get(&value)
            .copied()
            .unwrap_or(Variation::Varying)
    }

    /// The control variation governing a block, defaulting to `Varying` when
    /// the block is unknown or unreachable from the function entry.
    pub fn block_control(&self, block: BlockId) -> Variation {
        self.block_controls
            .get(&block)
            .copied()
            .unwrap_or(Variation::Varying)
    }

    /// Stable, source-order diagnostics produced by the analysis.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

/// A fail-closed analysis diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Diagnostic {
    /// A barrier is controlled by a value too varying for its participants.
    DivergentBarrier {
        block: BlockId,
        operation_index: usize,
        execution_scope: SynchronizationScope,
        control: Variation,
    },
    /// Current IR metadata is insufficient for a less conservative result.
    Unsupported {
        block: Option<BlockId>,
        operation_index: Option<usize>,
        reason: UnsupportedReason,
    },
}

/// Why the first analysis slice could not derive a stronger fact.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum UnsupportedReason {
    FunctionDeclaration,
    CallWithoutSummary {
        callee: FunctionId,
    },
    MalformedControlFlow,
    /// Postdominance was not established for these reachable blocks. This
    /// includes cycles whose termination/dynamic iteration count is unproved.
    PostdominanceUnavailable {
        blocks: Vec<BlockId>,
    },
    UnknownValue {
        value: ValueId,
    },
}

#[cfg(test)]
mod tests {
    use super::Variation;
    use fe2o3_kernel_ir::SynchronizationScope;

    #[test]
    fn lattice_join_selects_least_upper_bound() {
        let levels = [
            Variation::GridUniform,
            Variation::WorkgroupUniform,
            Variation::SubgroupUniform,
            Variation::Varying,
        ];

        for (left_index, left) in levels.into_iter().enumerate() {
            for (right_index, right) in levels.into_iter().enumerate() {
                assert_eq!(left.join(right), levels[left_index.max(right_index)]);
                assert_eq!(left.join(right), right.join(left));
            }
        }
    }

    #[test]
    fn scope_uniformity_follows_lattice_thresholds() {
        assert!(Variation::Varying.is_uniform_for(SynchronizationScope::Invocation));
        assert!(Variation::WorkgroupUniform.is_uniform_for(SynchronizationScope::Workgroup));
        assert!(!Variation::SubgroupUniform.is_uniform_for(SynchronizationScope::Workgroup));
        assert!(Variation::GridUniform.is_uniform_for(SynchronizationScope::Device));
        assert!(!Variation::WorkgroupUniform.is_uniform_for(SynchronizationScope::Device));
    }
}

mod canonical_ranked_view_v1;
pub use canonical_ranked_view_v1::*;

mod canonical_kir_aggregate_ssa_v18;
pub use canonical_kir_aggregate_ssa_v18::*;
mod source_ssa_boundaries_v31;
pub use source_ssa_boundaries_v31::*;
