//! Non-authoritative GPU group snapshots.
//!
//! Values in this module support checked rank and size arithmetic over
//! caller-asserted invocation and wave snapshots. They do not authenticate a
//! launch, target, current hardware state, control-flow epoch, or compiled
//! artifact. The current compiler does not construct these snapshots or lower
//! their synchronization entry point.

use core::fmt;
use core::marker::PhantomData;

use crate::sync;
use crate::thread::Invocation3D;
use crate::wave::{Wave64, WaveLane};

/// Version of the typed group and universal arithmetic contract.
pub const TYPED_GROUP_CONTRACT_VERSION_V1: u16 = 1;

mod sealed {
    pub trait Group {}
    pub trait SynchronizationContract {}
    pub trait Wave64TileWidth {}
}

/// Execution and memory scopes named by a group synchronization contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GroupScope {
    /// Every work-item in a launch grid.
    Grid,
    /// Every work-item in one workgroup.
    Workgroup,
    /// Participating lanes in one physical wave.
    Subgroup,
}

/// Memory ordering established by a supported group synchronization operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GroupMemoryOrdering {
    /// Writes before the operation are released and reads after it acquire them.
    AcquireRelease,
}

/// Address spaces ordered by a supported group synchronization operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GroupMemorySpace {
    /// Global memory observed by members of one workgroup.
    Global,
    /// AMD local data share visible within one workgroup.
    Workgroup,
}

/// Sealed type-level description of a group's synchronization semantics.
///
/// This is descriptive metadata, not convergence or execution authority. A
/// contract with `SUPPORTED == false` deliberately exposes no barrier method.
/// A supported contract records the execution scope, memory scope, ordering,
/// address spaces, and convergence requirement that compiler IR must preserve.
pub trait SynchronizationContract: sealed::SynchronizationContract {
    /// Whether this source group has a modeled synchronization operation.
    const SUPPORTED: bool;
    /// Participants that must reach the same dynamic synchronization operation.
    const EXECUTION_SCOPE: Option<GroupScope>;
    /// Visibility scope of memory ordered by the operation.
    const MEMORY_SCOPE: Option<GroupScope>;
    /// Ordering established by the operation.
    const ORDERING: Option<GroupMemoryOrdering>;
    /// Address spaces ordered by the operation.
    const ADDRESS_SPACES: &'static [GroupMemorySpace];
    /// Whether uniform arrival by the execution scope must be proven.
    const REQUIRES_UNIFORM_CONVERGENCE: bool;
}

/// Marker for a group whose synchronization behavior is intentionally absent.
#[derive(Debug)]
pub enum UnsupportedSynchronization {}

impl sealed::SynchronizationContract for UnsupportedSynchronization {}

impl SynchronizationContract for UnsupportedSynchronization {
    const SUPPORTED: bool = false;
    const EXECUTION_SCOPE: Option<GroupScope> = None;
    const MEMORY_SCOPE: Option<GroupScope> = None;
    const ORDERING: Option<GroupMemoryOrdering> = None;
    const ADDRESS_SPACES: &'static [GroupMemorySpace] = &[];
    const REQUIRES_UNIFORM_CONVERGENCE: bool = false;
}

/// Marker for CUDA-compatible workgroup synchronization semantics.
#[derive(Debug)]
pub enum WorkgroupSynchronization {}

impl sealed::SynchronizationContract for WorkgroupSynchronization {}

impl SynchronizationContract for WorkgroupSynchronization {
    const SUPPORTED: bool = true;
    const EXECUTION_SCOPE: Option<GroupScope> = Some(GroupScope::Workgroup);
    const MEMORY_SCOPE: Option<GroupScope> = Some(GroupScope::Workgroup);
    const ORDERING: Option<GroupMemoryOrdering> = Some(GroupMemoryOrdering::AcquireRelease);
    const ADDRESS_SPACES: &'static [GroupMemorySpace] =
        &[GroupMemorySpace::Global, GroupMemorySpace::Workgroup];
    const REQUIRES_UNIFORM_CONVERGENCE: bool = true;
}

/// Universal arithmetic interface implemented by every group snapshot.
///
/// Sizes and ranks use `u64` so the same generic algorithm can inspect grid,
/// workgroup, subgroup-tile, and active-lane snapshots without truncation. This
/// trait grants no execution, synchronization, memory, or launch authority.
#[rustc_diagnostic_item = "fe2o3_device_group_v1"]
pub trait Group: sealed::Group {
    /// Static synchronization policy for this group kind.
    type Synchronization: SynchronizationContract;

    /// Execution scope represented by this group kind.
    const SCOPE: GroupScope;

    /// Number of work-items or lanes in this exact group.
    fn size(&self) -> u64;

    /// Zero-based rank of the current invocation within this exact group.
    fn thread_rank(&self) -> u64;

    /// Whether the arithmetic snapshot has a rank in its declared extent.
    fn has_valid_rank(&self) -> bool {
        self.thread_rank() < self.size()
    }
}

/// Arithmetic snapshot of a launch grid and one invocation's rank within it.
///
/// Construction borrows a caller-asserted [`Invocation3D`] snapshot and fails
/// when the grid's total size or row-major rank cannot be represented in `u64`.
/// It does not establish that the snapshot describes the current hardware
/// invocation or epoch. This value is deliberately neither `Copy`, `Clone`,
/// `Send`, nor `Sync`.
#[rustc_diagnostic_item = "fe2o3_device_grid_group_v1"]
pub struct Grid<'invocation> {
    size: u64,
    thread_rank: u64,
    _invocation: PhantomData<&'invocation Invocation3D>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'invocation> Grid<'invocation> {
    /// Derives checked arithmetic from a caller-asserted invocation snapshot.
    #[rustc_diagnostic_item = "fe2o3_device_grid_from_invocation_snapshot_v1"]
    pub fn from_invocation_snapshot(invocation: &'invocation Invocation3D) -> Option<Self> {
        let extent = invocation.global_grid_size();
        Some(Self {
            size: extent.volume()?,
            thread_rank: invocation.global_workitem_id().linear(extent)?,
            _invocation: PhantomData,
            _not_send_sync: PhantomData,
        })
    }
}

impl fmt::Debug for Grid<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Grid")
            .field("size", &self.size)
            .field("thread_rank", &self.thread_rank)
            .finish()
    }
}

impl sealed::Group for Grid<'_> {}

impl Group for Grid<'_> {
    type Synchronization = UnsupportedSynchronization;

    const SCOPE: GroupScope = GroupScope::Grid;

    fn size(&self) -> u64 {
        self.size
    }

    fn thread_rank(&self) -> u64 {
        self.thread_rank
    }
}

/// Arithmetic snapshot of a workgroup and one invocation's rank within it.
///
/// Construction borrows a caller-asserted [`Invocation3D`] snapshot and fails
/// when the workgroup's total size or row-major rank cannot be represented in
/// `u64`. It does not establish that the snapshot describes the current
/// hardware invocation or epoch. This value is deliberately neither `Copy`,
/// `Clone`, `Send`, nor `Sync`.
#[rustc_diagnostic_item = "fe2o3_device_workgroup_group_v1"]
pub struct Workgroup<'invocation> {
    size: u64,
    thread_rank: u64,
    _invocation: PhantomData<&'invocation Invocation3D>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'invocation> Workgroup<'invocation> {
    /// Derives checked arithmetic from a caller-asserted invocation snapshot.
    #[rustc_diagnostic_item = "fe2o3_device_workgroup_from_invocation_snapshot_v1"]
    pub fn from_invocation_snapshot(invocation: &'invocation Invocation3D) -> Option<Self> {
        let size = invocation.workgroup_size();
        let id = invocation.workitem_id();
        let thread_rank = linear_rank_3d(
            u64::from(id.x()),
            u64::from(id.y()),
            u64::from(id.z()),
            u64::from(size.x()),
            u64::from(size.y()),
        );
        Some(Self {
            size: size.volume(),
            thread_rank,
            _invocation: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Executes one CUDA-compatible workgroup barrier.
    ///
    /// The compiler must prove that this arithmetic snapshot still describes
    /// the calling invocation's current workgroup. Every active work-item must
    /// execute this exact dynamic call once and in the same barrier sequence. No
    /// participating work-item may take a conditional path that skips the call,
    /// return, panic, or otherwise exit early. The compiler must preserve all
    /// semantics in [`WorkgroupSynchronization`], including workgroup-uniform
    /// convergence and acquire-release visibility for global and workgroup
    /// memory. Unsupported lowering rejects the call or retains the panic stub.
    #[rustc_diagnostic_item = "fe2o3_device_workgroup_synchronize_v1"]
    pub fn synchronize(&self) {
        sync::syncthreads()
    }
}

impl fmt::Debug for Workgroup<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Workgroup")
            .field("size", &self.size)
            .field("thread_rank", &self.thread_rank)
            .finish()
    }
}

impl sealed::Group for Workgroup<'_> {}

impl Group for Workgroup<'_> {
    type Synchronization = WorkgroupSynchronization;

    const SCOPE: GroupScope = GroupScope::Workgroup;

    fn size(&self) -> u64 {
        self.size
    }

    fn thread_rank(&self) -> u64 {
        self.thread_rank
    }
}

/// Type-level candidate for one arithmetic wave64 tile width.
///
/// [`ValidWave64TileWidth`] is implemented only for `1`, `2`, `4`, `8`, `16`,
/// `32`, and `64`: the power-of-two divisors of 64. This marker contains no
/// target or wave-mode authority.
#[derive(Debug)]
pub struct Wave64TileWidth<const N: u32>;

/// Sealed arithmetic proof that a const tile width partitions 64 lanes.
pub trait ValidWave64TileWidth: sealed::Wave64TileWidth {}

macro_rules! valid_wave64_tile_widths {
    ($($width:literal),+ $(,)?) => {
        $(
            impl sealed::Wave64TileWidth for Wave64TileWidth<$width> {}
            impl ValidWave64TileWidth for Wave64TileWidth<$width> {}
        )+
    };
}

valid_wave64_tile_widths!(1, 2, 4, 8, 16, 32, 64);

/// Arithmetic snapshot of a static contiguous tile in a wave64 snapshot.
///
/// `N` must be a supported power-of-two divisor of 64. The lifetime prevents
/// this value from outliving its caller-asserted lane snapshot, but does not
/// bind a hardware execution epoch. This value is deliberately neither `Copy`,
/// `Clone`, `Send`, nor `Sync`.
#[rustc_diagnostic_item = "fe2o3_device_subgroup_tile_v1"]
pub struct SubgroupTile<'wave, const N: u32>
where
    Wave64TileWidth<N>: ValidWave64TileWidth,
{
    lane: u32,
    _wave_snapshot: PhantomData<&'wave WaveLane<Wave64>>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'wave, const N: u32> SubgroupTile<'wave, N>
where
    Wave64TileWidth<N>: ValidWave64TileWidth,
{
    /// Derives tile arithmetic from a caller-asserted wave64 lane snapshot.
    #[rustc_diagnostic_item = "fe2o3_device_subgroup_tile_from_wave64_v1"]
    pub fn from_wave64_snapshot(lane: &'wave WaveLane<Wave64>) -> Self {
        Self {
            lane: lane.get(),
            _wave_snapshot: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Zero-based index of this tile within its physical wave.
    pub const fn tile_index(&self) -> u32 {
        self.lane / N
    }
}

impl<const N: u32> fmt::Debug for SubgroupTile<'_, N>
where
    Wave64TileWidth<N>: ValidWave64TileWidth,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SubgroupTile")
            .field("width", &N)
            .field("tile_index", &self.tile_index())
            .field("thread_rank", &(self.lane % N))
            .finish()
    }
}

impl<const N: u32> sealed::Group for SubgroupTile<'_, N> where
    Wave64TileWidth<N>: ValidWave64TileWidth
{
}

impl<const N: u32> Group for SubgroupTile<'_, N>
where
    Wave64TileWidth<N>: ValidWave64TileWidth,
{
    type Synchronization = UnsupportedSynchronization;

    const SCOPE: GroupScope = GroupScope::Subgroup;

    fn size(&self) -> u64 {
        N as u64
    }

    fn thread_rank(&self) -> u64 {
        (self.lane % N) as u64
    }
}

/// Arithmetic snapshot of a caller-asserted active-lane mask.
///
/// The lifetime prevents this value from outliving its lane snapshot, but it
/// does not bind a hardware execution epoch. In particular, this value is not
/// persistent EXEC authority and cannot authorize a collective or barrier.
/// This value is deliberately neither `Copy`, `Clone`, `Send`, nor `Sync`.
#[rustc_diagnostic_item = "fe2o3_device_active_lane_group_v1"]
pub struct ActiveLaneGroup<'wave> {
    lane: u32,
    asserted_mask: u64,
    _wave_snapshot: PhantomData<&'wave WaveLane<Wave64>>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'wave> ActiveLaneGroup<'wave> {
    /// Records a caller-asserted active mask for rank and size arithmetic.
    ///
    /// Returns `None` when `asserted_mask` excludes the lane snapshot.
    ///
    /// # Safety
    ///
    /// `asserted_mask` must equal the wave64 active mask observed for the same
    /// invocation at the source point represented by `lane`. This assertion is
    /// valid only for the arithmetic snapshot returned here; it grants no
    /// continuing EXEC, convergence, collective, synchronization, target, or
    /// epoch authority. The current compiler provides no checked constructor.
    #[rustc_diagnostic_item = "fe2o3_device_active_lane_group_from_snapshot_v1"]
    pub unsafe fn from_caller_asserted_snapshot(
        lane: &'wave WaveLane<Wave64>,
        asserted_mask: u64,
    ) -> Option<Self> {
        Self::checked_snapshot(lane, asserted_mask)
    }

    #[cfg(test)]
    // Validates modeled mask arithmetic without asserting an EXEC observation.
    fn from_model_snapshot(lane: &'wave WaveLane<Wave64>, model_mask: u64) -> Option<Self> {
        Self::checked_snapshot(lane, model_mask)
    }

    fn checked_snapshot(lane: &'wave WaveLane<Wave64>, mask: u64) -> Option<Self> {
        if mask & (1_u64 << lane.get()) == 0 {
            return None;
        }
        Some(Self {
            lane: lane.get(),
            asserted_mask: mask,
            _wave_snapshot: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Returns the caller-asserted mask as non-authoritative snapshot data.
    pub const fn caller_asserted_mask(&self) -> u64 {
        self.asserted_mask
    }
}

impl fmt::Debug for ActiveLaneGroup<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ActiveLaneGroup")
            .field(
                "caller_asserted_mask",
                &format_args!("{:#018x}", self.asserted_mask),
            )
            .field("thread_rank", &self.thread_rank())
            .finish()
    }
}

impl sealed::Group for ActiveLaneGroup<'_> {}

impl Group for ActiveLaneGroup<'_> {
    type Synchronization = UnsupportedSynchronization;

    const SCOPE: GroupScope = GroupScope::Subgroup;

    fn size(&self) -> u64 {
        u64::from(self.asserted_mask.count_ones())
    }

    fn thread_rank(&self) -> u64 {
        let lower_lanes = (1_u64 << self.lane) - 1;
        u64::from((self.asserted_mask & lower_lanes).count_ones())
    }
}

const fn linear_rank_3d(x: u64, y: u64, z: u64, extent_x: u64, extent_y: u64) -> u64 {
    (z * extent_y + y) * extent_x + x
}

#[cfg(test)]
mod tests;
