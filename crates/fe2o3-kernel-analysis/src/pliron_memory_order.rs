//! Workload-neutral memory-version and happens-before analysis for PLIRON.
//!
//! Facts come only from executed access, barrier, fence, and atomic events.
//! An ordering attribute is never treated as proof of a read-from edge.

use std::collections::{BTreeMap, HashMap};

use crate::pliron_invocation_trace::{
    PlironInvocationTraceV1, PlironTraceEventV1, PlironTraceLocationV1,
};
use crate::pliron_provenance_alias::PlironProvenanceAliasAnalysisV1;
use dialect_gpu::{
    AddressSpaceAttr, HierarchyAttr, MemoryOrderAttr as GpuMemoryOrderAttr, MemoryScopeAttr,
};
use dialect_kernel::{AccessKindAttr, AtomicOrderingAttr, AtomicScopeAttr, MemorySpaceAttr};

pub const MAX_PLIRON_MEMORY_VERSIONS_V1: usize = 1_048_576;
pub const MAX_PLIRON_MEMORY_ORDER_ISSUES_V1: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlironMemoryLocationV1 {
    block: usize,
    operation: usize,
}

impl From<PlironTraceLocationV1> for PlironMemoryLocationV1 {
    fn from(location: PlironTraceLocationV1) -> Self {
        Self {
            block: location.block,
            operation: location.operation,
        }
    }
}

impl PlironMemoryLocationV1 {
    pub const fn block(self) -> usize {
        self.block
    }

    pub const fn operation(self) -> usize {
        self.operation
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PlironMemoryAddressV1 {
    allocation_class: u64,
    indices: Vec<u64>,
}

impl PlironMemoryAddressV1 {
    pub const fn allocation_class(&self) -> u64 {
        self.allocation_class
    }

    pub fn indices(&self) -> &[u64] {
        &self.indices
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironMemoryVersionV1 {
    id: usize,
    grid: u64,
    workgroup: u64,
    epoch: usize,
    invocation: Vec<u64>,
    location: PlironMemoryLocationV1,
    address: PlironMemoryAddressV1,
    access: AccessKindAttr,
}

impl PlironMemoryVersionV1 {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub const fn epoch(&self) -> usize {
        self.epoch
    }

    pub const fn grid(&self) -> u64 {
        self.grid
    }

    pub const fn workgroup(&self) -> u64 {
        self.workgroup
    }

    pub fn invocation(&self) -> &[u64] {
        &self.invocation
    }

    pub const fn location(&self) -> PlironMemoryLocationV1 {
        self.location
    }

    pub const fn address(&self) -> &PlironMemoryAddressV1 {
        &self.address
    }

    pub const fn access(&self) -> AccessKindAttr {
        self.access
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironPublicationEdgeV1 {
    version: usize,
    barrier: PlironMemoryLocationV1,
    reader_invocation: Vec<u64>,
    read: PlironMemoryLocationV1,
}

impl PlironPublicationEdgeV1 {
    pub const fn version(&self) -> usize {
        self.version
    }

    pub const fn barrier(&self) -> PlironMemoryLocationV1 {
        self.barrier
    }

    pub fn reader_invocation(&self) -> &[u64] {
        &self.reader_invocation
    }

    pub const fn read(&self) -> PlironMemoryLocationV1 {
        self.read
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironMemoryOrderIssueV1 {
    ReadBeforeInitialization {
        invocation: Vec<u64>,
        location: PlironMemoryLocationV1,
        address: PlironMemoryAddressV1,
    },
    ConflictingEffects {
        address: PlironMemoryAddressV1,
        first_invocation: Vec<u64>,
        first_location: PlironMemoryLocationV1,
        first_access: AccessKindAttr,
        second_invocation: Vec<u64>,
        second_location: PlironMemoryLocationV1,
        second_access: AccessKindAttr,
    },
    AtomicReadFromUnresolved {
        invocation: Vec<u64>,
        location: PlironMemoryLocationV1,
        address: PlironMemoryAddressV1,
        detail: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironMemoryOrderFailureV1 {
    UnresolvedAddress {
        location: PlironMemoryLocationV1,
    },
    MismatchedBarrierPhase {
        grid: u64,
        workgroup: u64,
        epoch: usize,
    },
    SubgroupPublicationUnsupported {
        location: PlironMemoryLocationV1,
    },
    FencePublicationUnsupported {
        location: PlironMemoryLocationV1,
    },
    VersionLimitExceeded,
    IssueLimitExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironMemoryOrderAnalysisV1 {
    versions: Vec<PlironMemoryVersionV1>,
    publication_edges: Vec<PlironPublicationEdgeV1>,
    issues: Vec<PlironMemoryOrderIssueV1>,
}

impl PlironMemoryOrderAnalysisV1 {
    pub fn versions(&self) -> &[PlironMemoryVersionV1] {
        &self.versions
    }

    pub fn publication_edges(&self) -> &[PlironPublicationEdgeV1] {
        &self.publication_edges
    }

    pub fn issues(&self) -> &[PlironMemoryOrderIssueV1] {
        &self.issues
    }
}

#[derive(Clone)]
struct EffectV1 {
    invocation: Vec<u64>,
    location: PlironMemoryLocationV1,
    access: AccessKindAttr,
}

#[derive(Clone)]
struct ReleaseAtomicV1 {
    invocation: Vec<u64>,
    event: usize,
    scope: AtomicScopeAttr,
}

#[derive(Default)]
struct EpochAddressStateV1 {
    effects: Vec<EffectV1>,
}

#[derive(Clone, Debug)]
struct PublishedVersionV1 {
    id: usize,
    barrier: PlironMemoryLocationV1,
}

pub(crate) fn analyze_pliron_memory_order_v1(
    traces: &[PlironInvocationTraceV1],
    provenance: &PlironProvenanceAliasAnalysisV1,
) -> Result<PlironMemoryOrderAnalysisV1, PlironMemoryOrderFailureV1> {
    if let Some(location) = traces
        .iter()
        .flat_map(|trace| &trace.events)
        .find_map(|event| match event {
            PlironTraceEventV1::Fence {
                location,
                address_space: AddressSpaceAttr::Workgroup,
                ..
            } => Some((*location).into()),
            _ => None,
        })
    {
        return Err(PlironMemoryOrderFailureV1::FencePublicationUnsupported { location });
    }
    let mut grouped = BTreeMap::<(u64, u64), Vec<&PlironInvocationTraceV1>>::new();
    for trace in traces {
        grouped
            .entry((trace.grid, trace.workgroup))
            .or_default()
            .push(trace);
    }

    let mut analysis = PlironMemoryOrderAnalysisV1 {
        versions: Vec::new(),
        publication_edges: Vec::new(),
        issues: Vec::new(),
    };
    for ((grid, workgroup), group) in grouped {
        analyze_workgroup(grid, workgroup, &group, provenance, &mut analysis)?;
    }
    Ok(analysis)
}

fn analyze_workgroup(
    grid: u64,
    workgroup: u64,
    traces: &[&PlironInvocationTraceV1],
    provenance: &PlironProvenanceAliasAnalysisV1,
    analysis: &mut PlironMemoryOrderAnalysisV1,
) -> Result<(), PlironMemoryOrderFailureV1> {
    let mut cursors = vec![0_usize; traces.len()];
    let mut epoch = 0_usize;
    let mut published = HashMap::<PlironMemoryAddressV1, Vec<PublishedVersionV1>>::new();
    let release_atomics = collect_release_atomics(traces, provenance);

    loop {
        let mut epoch_states = HashMap::<PlironMemoryAddressV1, EpochAddressStateV1>::new();
        let mut epoch_versions = HashMap::<PlironMemoryAddressV1, Vec<usize>>::new();
        let mut local_versions =
            vec![HashMap::<PlironMemoryAddressV1, Vec<usize>>::new(); traces.len()];
        let mut barriers = Vec::new();
        let mut any_event = false;

        for (trace_index, trace) in traces.iter().enumerate() {
            while let Some(event) = trace.events.get(cursors[trace_index]) {
                cursors[trace_index] += 1;
                any_event = true;
                match event {
                    PlironTraceEventV1::Barrier {
                        location,
                        execution_scope: HierarchyAttr::Workgroup,
                        memory_scope:
                            MemoryScopeAttr::Workgroup
                            | MemoryScopeAttr::Device
                            | MemoryScopeAttr::System,
                        address_space: AddressSpaceAttr::Workgroup,
                        order:
                            GpuMemoryOrderAttr::AcquireRelease
                            | GpuMemoryOrderAttr::SequentiallyConsistent,
                        ..
                    } => {
                        barriers.push((*location).into());
                        break;
                    }
                    PlironTraceEventV1::Barrier {
                        location,
                        execution_scope: HierarchyAttr::Subgroup,
                        address_space: AddressSpaceAttr::Workgroup,
                        ..
                    } => {
                        return Err(PlironMemoryOrderFailureV1::SubgroupPublicationUnsupported {
                            location: (*location).into(),
                        });
                    }
                    PlironTraceEventV1::Memory {
                        location,
                        memory_space: MemorySpaceAttr::Workgroup,
                        access,
                        indices,
                        noalias_class,
                        ..
                    } => {
                        let location = (*location).into();
                        let Some(indices) = indices.iter().copied().collect::<Option<Vec<_>>>()
                        else {
                            return Err(PlironMemoryOrderFailureV1::UnresolvedAddress { location });
                        };
                        let address = PlironMemoryAddressV1 {
                            allocation_class: provenance
                                .canonical_class(MemorySpaceAttr::Workgroup, *noalias_class),
                            indices,
                        };
                        let effect = EffectV1 {
                            invocation: trace.invocation.clone(),
                            location,
                            access: *access,
                        };

                        if access.reads_memory()
                            && !local_versions[trace_index].contains_key(&address)
                            && !published.contains_key(&address)
                        {
                            let issue = if has_plausible_atomic_publication(
                                trace,
                                cursors[trace_index],
                                provenance,
                                &release_atomics,
                            ) {
                                PlironMemoryOrderIssueV1::AtomicReadFromUnresolved {
                                    invocation: trace.invocation.clone(),
                                    location,
                                    address: address.clone(),
                                    detail: "an acquire/release declaration does not identify the write observed by this read".to_owned(),
                                }
                            } else {
                                PlironMemoryOrderIssueV1::ReadBeforeInitialization {
                                    invocation: trace.invocation.clone(),
                                    location,
                                    address: address.clone(),
                                }
                            };
                            push_issue(analysis, issue)?;
                        }

                        if let Some(state) = epoch_states.get(&address)
                            && let Some(first) = state.effects.iter().find(|first| {
                                first.invocation != effect.invocation
                                    && effects_conflict(first.access, effect.access)
                            })
                        {
                            push_issue(
                                analysis,
                                PlironMemoryOrderIssueV1::ConflictingEffects {
                                    address: address.clone(),
                                    first_invocation: first.invocation.clone(),
                                    first_location: first.location,
                                    first_access: first.access,
                                    second_invocation: effect.invocation.clone(),
                                    second_location: effect.location,
                                    second_access: effect.access,
                                },
                            )?;
                        }
                        epoch_states
                            .entry(address.clone())
                            .or_default()
                            .effects
                            .push(effect);

                        if access.reads_memory()
                            && !local_versions[trace_index].contains_key(&address)
                            && let Some(visible) = published.get(&address)
                        {
                            for version in visible {
                                analysis.publication_edges.push(PlironPublicationEdgeV1 {
                                    version: version.id,
                                    barrier: version.barrier,
                                    reader_invocation: trace.invocation.clone(),
                                    read: location,
                                });
                            }
                        }
                        if access.writes_memory() {
                            if analysis.versions.len() == MAX_PLIRON_MEMORY_VERSIONS_V1 {
                                return Err(PlironMemoryOrderFailureV1::VersionLimitExceeded);
                            }
                            let id = analysis.versions.len();
                            analysis.versions.push(PlironMemoryVersionV1 {
                                id,
                                grid,
                                workgroup,
                                epoch,
                                invocation: trace.invocation.clone(),
                                location,
                                address: address.clone(),
                                access: *access,
                            });
                            epoch_versions.entry(address.clone()).or_default().push(id);
                            local_versions[trace_index]
                                .entry(address)
                                .or_default()
                                .push(id);
                        }
                    }
                    PlironTraceEventV1::Barrier { .. }
                    | PlironTraceEventV1::Fence { .. }
                    | PlironTraceEventV1::TensorInstruction { .. }
                    | PlironTraceEventV1::Trap { .. }
                    | PlironTraceEventV1::Memory { .. }
                    | PlironTraceEventV1::CollectiveAllocation { .. } => {}
                }
            }
        }

        if !any_event {
            break;
        }
        if !barriers.is_empty() {
            if barriers.len() != traces.len()
                || barriers.iter().any(|barrier| *barrier != barriers[0])
            {
                return Err(PlironMemoryOrderFailureV1::MismatchedBarrierPhase {
                    grid,
                    workgroup,
                    epoch,
                });
            }
            for (address, versions) in epoch_versions {
                published.insert(
                    address,
                    versions
                        .into_iter()
                        .map(|id| PublishedVersionV1 {
                            id,
                            barrier: barriers[0],
                        })
                        .collect(),
                );
            }
            epoch = epoch.saturating_add(1);
        }
        if cursors
            .iter()
            .zip(traces)
            .all(|(cursor, trace)| *cursor == trace.events.len())
        {
            break;
        }
    }
    Ok(())
}

fn effects_conflict(first: AccessKindAttr, second: AccessKindAttr) -> bool {
    if !first.writes_memory() && !second.writes_memory() {
        return false;
    }
    !(first.is_atomic() && second.is_atomic())
}

fn collect_release_atomics(
    traces: &[&PlironInvocationTraceV1],
    provenance: &PlironProvenanceAliasAnalysisV1,
) -> HashMap<PlironMemoryAddressV1, Vec<ReleaseAtomicV1>> {
    let mut releases = HashMap::<PlironMemoryAddressV1, Vec<ReleaseAtomicV1>>::new();
    for trace in traces {
        for (event_index, event) in trace.events.iter().enumerate() {
            let PlironTraceEventV1::Memory {
                memory_space: MemorySpaceAttr::Workgroup,
                access,
                atomic_ordering:
                    Some(
                        AtomicOrderingAttr::Release
                        | AtomicOrderingAttr::AcquireRelease
                        | AtomicOrderingAttr::SequentiallyConsistent,
                    ),
                atomic_scope: Some(scope),
                indices,
                noalias_class,
                ..
            } = event
            else {
                continue;
            };
            if !access.is_atomic()
                || !access.writes_memory()
                || scope.rank() < AtomicScopeAttr::Workgroup.rank()
            {
                continue;
            }
            let Some(indices) = indices.iter().copied().collect::<Option<Vec<_>>>() else {
                continue;
            };
            releases
                .entry(PlironMemoryAddressV1 {
                    allocation_class: provenance
                        .canonical_class(MemorySpaceAttr::Workgroup, *noalias_class),
                    indices,
                })
                .or_default()
                .push(ReleaseAtomicV1 {
                    invocation: trace.invocation.clone(),
                    event: event_index,
                    scope: *scope,
                });
        }
    }
    releases
}

fn has_plausible_atomic_publication(
    trace: &PlironInvocationTraceV1,
    cursor: usize,
    provenance: &PlironProvenanceAliasAnalysisV1,
    releases: &HashMap<PlironMemoryAddressV1, Vec<ReleaseAtomicV1>>,
) -> bool {
    trace.events[..cursor]
        .iter()
        .enumerate()
        .any(|(acquire_event, event)| {
            let PlironTraceEventV1::Memory {
                memory_space: MemorySpaceAttr::Workgroup,
                access,
                atomic_ordering:
                    Some(
                        AtomicOrderingAttr::Acquire
                        | AtomicOrderingAttr::AcquireRelease
                        | AtomicOrderingAttr::SequentiallyConsistent,
                    ),
                atomic_scope: Some(acquire_scope),
                indices,
                noalias_class,
                ..
            } = event
            else {
                return false;
            };
            if !access.is_atomic()
                || !access.reads_memory()
                || acquire_scope.rank() < AtomicScopeAttr::Workgroup.rank()
            {
                return false;
            }
            let Some(indices) = indices.iter().copied().collect::<Option<Vec<_>>>() else {
                return false;
            };
            let address = PlironMemoryAddressV1 {
                allocation_class: provenance
                    .canonical_class(MemorySpaceAttr::Workgroup, *noalias_class),
                indices,
            };
            releases.get(&address).is_some_and(|candidates| {
                candidates.iter().any(|release| {
                    release.scope.rank() >= AtomicScopeAttr::Workgroup.rank()
                        && (release.invocation != trace.invocation || release.event < acquire_event)
                })
            })
        })
}

fn push_issue(
    analysis: &mut PlironMemoryOrderAnalysisV1,
    issue: PlironMemoryOrderIssueV1,
) -> Result<(), PlironMemoryOrderFailureV1> {
    if analysis.issues.len() == MAX_PLIRON_MEMORY_ORDER_ISSUES_V1 {
        return Err(PlironMemoryOrderFailureV1::IssueLimitExceeded);
    }
    analysis.issues.push(issue);
    Ok(())
}
