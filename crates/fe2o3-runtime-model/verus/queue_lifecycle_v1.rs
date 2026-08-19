use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub struct DeviceKeyV1 {
    pub physical: nat,
    pub generation: nat,
}

#[derive(PartialEq, Eq)]
pub struct VmKeyV1 {
    pub device: DeviceKeyV1,
    pub id: nat,
}

#[derive(PartialEq, Eq)]
pub struct MappingKeyV1 {
    pub vm: VmKeyV1,
    pub allocation_id: nat,
    pub allocation_generation: nat,
    pub mapping_id: nat,
}

#[derive(PartialEq, Eq)]
pub enum QueueResourceRoleV1 {
    Ring,
    Control,
    EndOfPipe,
    ContextSave,
}

pub struct QueueResourceV1 {
    pub role: QueueResourceRoleV1,
    pub mapping: MappingKeyV1,
    pub publication_id: nat,
    pub expected_memory_kind: nat,
    pub expected_coherence: nat,
    pub expected_access: nat,
}

pub struct QueuePlanV1 {
    pub vm: VmKeyV1,
    pub plan_id: nat,
    pub queue_instance_id: nat,
    pub queue_generation: nat,
    pub configuration_id: nat,
    pub resources: Seq<QueueResourceV1>,
}

pub open spec fn queue_resources_distinct_v1(
    left: QueueResourceV1,
    right: QueueResourceV1,
) -> bool {
    &&& left.mapping.allocation_id != right.mapping.allocation_id
    &&& left.mapping != right.mapping
    &&& left.publication_id != right.publication_id
}

pub open spec fn canonical_queue_resources_v1(plan: QueuePlanV1) -> bool {
    &&& plan.vm.device.physical > 0
    &&& plan.vm.device.generation > 0
    &&& plan.vm.id > 0
    &&& plan.plan_id > 0
    &&& plan.queue_instance_id > 0
    &&& plan.queue_generation > 0
    &&& plan.configuration_id > 0
    &&& plan.resources.len() == 4
    &&& plan.resources[0].role == QueueResourceRoleV1::Ring
    &&& plan.resources[1].role == QueueResourceRoleV1::Control
    &&& plan.resources[2].role == QueueResourceRoleV1::EndOfPipe
    &&& plan.resources[3].role == QueueResourceRoleV1::ContextSave
    &&& forall |i: int| 0 <= i < plan.resources.len() ==> {
        &&& #[trigger] plan.resources[i].mapping.vm == plan.vm
        &&& plan.resources[i].mapping.allocation_id > 0
        &&& plan.resources[i].mapping.allocation_generation > 0
        &&& plan.resources[i].mapping.mapping_id > 0
        &&& plan.resources[i].publication_id > 0
    }
    &&& queue_resources_distinct_v1(plan.resources[0], plan.resources[1])
    &&& queue_resources_distinct_v1(plan.resources[0], plan.resources[2])
    &&& queue_resources_distinct_v1(plan.resources[0], plan.resources[3])
    &&& queue_resources_distinct_v1(plan.resources[1], plan.resources[2])
    &&& queue_resources_distinct_v1(plan.resources[1], plan.resources[3])
    &&& queue_resources_distinct_v1(plan.resources[2], plan.resources[3])
}

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    Planned,
    CreatePending,
    Active,
    UpdatePending,
    DisablePending,
    Disabled,
    DestroyPending,
    CancelledBeforeCreate,
    Destroyed,
    Ambiguous,
}

#[derive(PartialEq, Eq)]
pub enum QueueOperationV1 {
    Create,
    Update,
    Disable,
    Destroy,
}

#[derive(PartialEq, Eq)]
pub enum QueueStatusV1 {
    Succeeded,
    FailedNoEffect,
    Indeterminate,
}

pub struct QueueRecordV1 {
    pub plan: QueuePlanV1,
    pub phase: QueuePhaseV1,
    pub native_queue_id: Option<nat>,
    pub configuration_id: nat,
}

pub open spec fn queue_retains_resources_v1(phase: QueuePhaseV1) -> bool {
    phase != QueuePhaseV1::CancelledBeforeCreate && phase != QueuePhaseV1::Destroyed
}

pub open spec fn max_u32_queue_id_v1() -> nat {
    0xffff_ffff
}

#[derive(PartialEq, Eq)]
pub enum CreateQueueIdFieldV1 {
    SentinelUnchanged,
    Returned(nat),
}

pub open spec fn classify_create_queue_id_v1(
    field: CreateQueueIdFieldV1,
) -> Option<nat> {
    match field {
        CreateQueueIdFieldV1::Returned(queue_id) if queue_id < max_u32_queue_id_v1() => {
            Some(queue_id)
        },
        _ => None,
    }
}

pub open spec fn observe_create_v1(
    old: QueueRecordV1,
    status: QueueStatusV1,
    field: CreateQueueIdFieldV1,
    returned_id_collision: bool,
    unresolved_process_identity: bool,
) -> QueueRecordV1 {
    let classified = classify_create_queue_id_v1(field);
    if old.phase != QueuePhaseV1::CreatePending {
        old
    } else if status == QueueStatusV1::Succeeded
        && classified.is_some()
        && !returned_id_collision
        && !unresolved_process_identity
    {
        QueueRecordV1 {
            plan: old.plan,
            phase: QueuePhaseV1::Active,
            native_queue_id: classified,
            configuration_id: old.configuration_id,
        }
    } else if status == QueueStatusV1::FailedNoEffect
        && field == CreateQueueIdFieldV1::SentinelUnchanged
    {
        QueueRecordV1 {
            plan: old.plan,
            phase: QueuePhaseV1::Planned,
            native_queue_id: None,
            configuration_id: old.configuration_id,
        }
    } else {
        QueueRecordV1 {
            plan: old.plan,
            phase: QueuePhaseV1::Ambiguous,
            native_queue_id: if returned_id_collision || unresolved_process_identity {
                None
            } else {
                classified
            },
            configuration_id: old.configuration_id,
        }
    }
}

pub open spec fn legal_non_create_pending_operation_v1(
    operation: QueueOperationV1,
    phase: QueuePhaseV1,
) -> bool {
    match operation {
        QueueOperationV1::Create => false,
        QueueOperationV1::Update => phase == QueuePhaseV1::UpdatePending,
        QueueOperationV1::Disable => phase == QueuePhaseV1::DisablePending,
        QueueOperationV1::Destroy => phase == QueuePhaseV1::DestroyPending,
    }
}

pub open spec fn observe_non_create_indeterminate_v1(
    old: QueueRecordV1,
    operation: QueueOperationV1,
) -> QueueRecordV1 {
    if legal_non_create_pending_operation_v1(operation, old.phase) {
        QueueRecordV1 {
            plan: old.plan,
            phase: QueuePhaseV1::Ambiguous,
            native_queue_id: old.native_queue_id,
            configuration_id: old.configuration_id,
        }
    } else {
        old
    }
}

pub open spec fn cancel_plan_v1(old: QueueRecordV1) -> QueueRecordV1 {
    if old.phase == QueuePhaseV1::Planned {
        QueueRecordV1 {
            plan: old.plan,
            phase: QueuePhaseV1::CancelledBeforeCreate,
            native_queue_id: None,
            configuration_id: old.configuration_id,
        }
    } else {
        old
    }
}

pub open spec fn observe_destroy_success_v1(old: QueueRecordV1) -> QueueRecordV1 {
    if old.phase == QueuePhaseV1::DestroyPending {
        QueueRecordV1 {
            plan: old.plan,
            phase: QueuePhaseV1::Destroyed,
            native_queue_id: old.native_queue_id,
            configuration_id: old.configuration_id,
        }
    } else {
        old
    }
}

pub open spec fn plan_contains_mapping_v1(
    plan: QueuePlanV1,
    mapping: MappingKeyV1,
) -> bool {
    exists |i: int| 0 <= i < plan.resources.len()
        && #[trigger] plan.resources[i].mapping == mapping
}

pub open spec fn can_release_mapping_v1(
    mapping: MappingKeyV1,
    queues: Seq<QueueRecordV1>,
) -> bool {
    forall |i: int| 0 <= i < queues.len()
        && queue_retains_resources_v1(#[trigger] queues[i].phase)
        ==> !plan_contains_mapping_v1(queues[i].plan, mapping)
}

#[derive(PartialEq, Eq)]
pub enum PublicationOwnerV1 {
    Generic,
    ComputeAqlQueue {
        vm: VmKeyV1,
        queue_instance_id: nat,
        queue_generation: nat,
    },
}

pub struct PublicationLeaseV1 {
    pub mapping: MappingKeyV1,
    pub publication_id: nat,
    pub owner: PublicationOwnerV1,
    pub live: bool,
}

pub open spec fn queue_publication_v1(
    plan: QueuePlanV1,
    resource: QueueResourceV1,
) -> PublicationLeaseV1 {
    PublicationLeaseV1 {
        mapping: resource.mapping,
        publication_id: resource.publication_id,
        owner: PublicationOwnerV1::ComputeAqlQueue {
            vm: plan.vm,
            queue_instance_id: plan.queue_instance_id,
            queue_generation: plan.queue_generation,
        },
        live: true,
    }
}

pub open spec fn generic_release_publication_v1(
    lease: PublicationLeaseV1,
) -> Option<PublicationLeaseV1> {
    if lease.owner == PublicationOwnerV1::Generic && lease.live {
        Some(PublicationLeaseV1 {
            mapping: lease.mapping,
            publication_id: lease.publication_id,
            owner: lease.owner,
            live: false,
        })
    } else {
        None
    }
}

pub open spec fn known_queue_id_reserved_v1(record: QueueRecordV1, queue_id: nat) -> bool {
    &&& record.native_queue_id == Some(queue_id)
    &&& (record.phase == QueuePhaseV1::Active
        || record.phase == QueuePhaseV1::UpdatePending
        || record.phase == QueuePhaseV1::DisablePending
        || record.phase == QueuePhaseV1::Disabled
        || record.phase == QueuePhaseV1::DestroyPending
        || record.phase == QueuePhaseV1::Ambiguous)
}

pub open spec fn unresolved_queue_identity_v1(record: QueueRecordV1) -> bool {
    (record.phase == QueuePhaseV1::CreatePending
        || record.phase == QueuePhaseV1::Ambiguous)
        && record.native_queue_id.is_none()
}

pub open spec fn returned_queue_id_available_v1(
    queues: Seq<QueueRecordV1>,
    queue_id: nat,
) -> bool {
    forall |i: int| 0 <= i < queues.len()
        ==> !known_queue_id_reserved_v1(#[trigger] queues[i], queue_id)
}

pub open spec fn process_can_begin_create_v1(queues: Seq<QueueRecordV1>) -> bool {
    forall |i: int| 0 <= i < queues.len()
        ==> !unresolved_queue_identity_v1(#[trigger] queues[i])
}

pub open spec fn at_most_one_create_pending_v1(queues: Seq<QueueRecordV1>) -> bool {
    forall |left: int, right: int|
        0 <= left < queues.len()
            && 0 <= right < queues.len()
            && left != right
            ==> !(#[trigger] queues[left].phase == QueuePhaseV1::CreatePending
                && #[trigger] queues[right].phase == QueuePhaseV1::CreatePending)
}

pub struct QueueHistoryEntryV1 {
    pub sequence: nat,
    pub queue_instance_id: nat,
    pub queue_generation: nat,
    pub from: QueuePhaseV1,
    pub to: QueuePhaseV1,
    pub native_queue_id: Option<nat>,
    pub configuration_id: nat,
}

pub open spec fn append_history_v1(
    history: Seq<QueueHistoryEntryV1>,
    entry: QueueHistoryEntryV1,
) -> Seq<QueueHistoryEntryV1> {
    history.push(entry)
}

pub open spec fn example_vm_v1() -> VmKeyV1 {
    VmKeyV1 {
        device: DeviceKeyV1 { physical: 1, generation: 1 },
        id: 1,
    }
}

pub open spec fn example_resource_v1(
    role: QueueResourceRoleV1,
    identity: nat,
) -> QueueResourceV1 {
    QueueResourceV1 {
        role,
        mapping: MappingKeyV1 {
            vm: example_vm_v1(),
            allocation_id: identity,
            allocation_generation: 1,
            mapping_id: identity,
        },
        publication_id: identity,
        expected_memory_kind: 1,
        expected_coherence: 1,
        expected_access: 1,
    }
}

pub open spec fn example_plan_v1() -> QueuePlanV1 {
    QueuePlanV1 {
        vm: example_vm_v1(),
        plan_id: 1,
        queue_instance_id: 1,
        queue_generation: 1,
        configuration_id: 1,
        resources: Seq::empty()
            .push(example_resource_v1(QueueResourceRoleV1::Ring, 1))
            .push(example_resource_v1(QueueResourceRoleV1::Control, 2))
            .push(example_resource_v1(QueueResourceRoleV1::EndOfPipe, 3))
            .push(example_resource_v1(QueueResourceRoleV1::ContextSave, 4)),
    }
}

pub proof fn canonical_four_resource_plan_is_inhabited_v1()
    ensures
        canonical_queue_resources_v1(example_plan_v1()),
{
}

pub proof fn exact_four_resources_are_compositely_rooted_and_distinct_v1(plan: QueuePlanV1)
    requires
        canonical_queue_resources_v1(plan),
    ensures
        plan.resources.len() == 4,
        forall |i: int| 0 <= i < 4
            ==> #[trigger] plan.resources[i].mapping.vm == plan.vm,
        plan.resources[0].mapping != plan.resources[1].mapping,
        plan.resources[0].mapping != plan.resources[2].mapping,
        plan.resources[0].mapping != plan.resources[3].mapping,
        plan.resources[1].mapping != plan.resources[2].mapping,
        plan.resources[1].mapping != plan.resources[3].mapping,
        plan.resources[2].mapping != plan.resources[3].mapping,
{
}

pub proof fn successful_create_retains_plan_generation_and_configuration_v1(
    old: QueueRecordV1,
    native_queue_id: nat,
)
    requires
        canonical_queue_resources_v1(old.plan),
        old.phase == QueuePhaseV1::CreatePending,
        native_queue_id < max_u32_queue_id_v1(),
    ensures
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(native_queue_id),
            false,
            false,
        ).phase == QueuePhaseV1::Active,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(native_queue_id),
            false,
            false,
        ).native_queue_id == Some(native_queue_id),
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(native_queue_id),
            false,
            false,
        ).plan.resources =~= old.plan.resources,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(native_queue_id),
            false,
            false,
        ).plan.queue_generation == old.plan.queue_generation,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(native_queue_id),
            false,
            false,
        ).configuration_id == old.configuration_id,
{
}

pub proof fn create_field_status_and_collision_are_fail_closed_v1(old: QueueRecordV1)
    requires
        old.phase == QueuePhaseV1::CreatePending,
    ensures
        classify_create_queue_id_v1(
            CreateQueueIdFieldV1::Returned(max_u32_queue_id_v1()),
        ).is_none(),
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(max_u32_queue_id_v1()),
            false,
            false,
        ).phase == QueuePhaseV1::Ambiguous,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(max_u32_queue_id_v1()),
            false,
            false,
        ).native_queue_id.is_none(),
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::SentinelUnchanged,
            false,
            false,
        ).phase == QueuePhaseV1::Ambiguous,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(0),
            true,
            false,
        ).phase == QueuePhaseV1::Ambiguous,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(0),
            true,
            false,
        ).native_queue_id.is_none(),
        observe_create_v1(
            old,
            QueueStatusV1::Indeterminate,
            CreateQueueIdFieldV1::Returned(0),
            false,
            false,
        ).phase == QueuePhaseV1::Ambiguous,
        observe_create_v1(
            old,
            QueueStatusV1::Indeterminate,
            CreateQueueIdFieldV1::Returned(0),
            false,
            false,
        ).native_queue_id == Some(0),
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(0),
            false,
            false,
        ).phase == QueuePhaseV1::Active,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(0),
            false,
            true,
        ).phase == QueuePhaseV1::Ambiguous,
        observe_create_v1(
            old,
            QueueStatusV1::Succeeded,
            CreateQueueIdFieldV1::Returned(0),
            false,
            true,
        ).native_queue_id.is_none(),
{
}

pub proof fn legal_non_create_indeterminate_operation_is_ambiguous_and_retaining_v1(
    old: QueueRecordV1,
    operation: QueueOperationV1,
)
    requires
        canonical_queue_resources_v1(old.plan),
        legal_non_create_pending_operation_v1(operation, old.phase),
    ensures
        operation != QueueOperationV1::Create,
        observe_non_create_indeterminate_v1(old, operation).phase == QueuePhaseV1::Ambiguous,
        observe_non_create_indeterminate_v1(old, operation).plan.resources =~= old.plan.resources,
        observe_non_create_indeterminate_v1(old, operation).native_queue_id == old.native_queue_id,
        observe_non_create_indeterminate_v1(old, operation).configuration_id
            == old.configuration_id,
        queue_retains_resources_v1(
            observe_non_create_indeterminate_v1(old, operation).phase,
        ),
{
}

pub proof fn cancel_and_destroy_are_the_exact_nonretaining_terminals_v1(
    planned: QueueRecordV1,
    destroy_pending: QueueRecordV1,
    phase: QueuePhaseV1,
)
    requires
        planned.phase == QueuePhaseV1::Planned,
        destroy_pending.phase == QueuePhaseV1::DestroyPending,
    ensures
        !queue_retains_resources_v1(cancel_plan_v1(planned).phase),
        cancel_plan_v1(planned).phase == QueuePhaseV1::CancelledBeforeCreate,
        !queue_retains_resources_v1(observe_destroy_success_v1(destroy_pending).phase),
        observe_destroy_success_v1(destroy_pending).phase == QueuePhaseV1::Destroyed,
        !queue_retains_resources_v1(phase)
            <==> phase == QueuePhaseV1::CancelledBeforeCreate
                || phase == QueuePhaseV1::Destroyed,
{
}

pub proof fn queue_owned_publication_rejects_generic_release_v1(
    plan: QueuePlanV1,
    resource_index: int,
)
    requires
        canonical_queue_resources_v1(plan),
        0 <= resource_index < plan.resources.len(),
    ensures
        queue_publication_v1(plan, plan.resources[resource_index]).live,
        generic_release_publication_v1(
            queue_publication_v1(plan, plan.resources[resource_index]),
        ).is_none(),
{
}

pub proof fn any_retaining_queue_blocks_its_exact_composite_mapping_v1(
    mapping: MappingKeyV1,
    queues: Seq<QueueRecordV1>,
    retained_index: int,
)
    requires
        0 <= retained_index < queues.len(),
        queue_retains_resources_v1(queues[retained_index].phase),
        plan_contains_mapping_v1(queues[retained_index].plan, mapping),
    ensures
        !can_release_mapping_v1(mapping, queues),
{
}

pub proof fn ambiguous_known_id_and_unknown_id_exclude_future_create_v1(
    known: QueueRecordV1,
    unknown: QueueRecordV1,
    second_unknown: QueueRecordV1,
    pending: QueueRecordV1,
    queue_id: nat,
)
    requires
        known.phase == QueuePhaseV1::Ambiguous,
        known.native_queue_id == Some(queue_id),
        unknown.phase == QueuePhaseV1::Ambiguous,
        unknown.native_queue_id.is_none(),
        second_unknown.phase == QueuePhaseV1::Ambiguous,
        second_unknown.native_queue_id.is_none(),
        pending.phase == QueuePhaseV1::CreatePending,
        pending.native_queue_id.is_none(),
    ensures
        !returned_queue_id_available_v1(Seq::empty().push(known), queue_id),
        !process_can_begin_create_v1(Seq::empty().push(unknown)),
        !process_can_begin_create_v1(Seq::empty().push(unknown).push(second_unknown)),
        at_most_one_create_pending_v1(Seq::empty().push(unknown).push(second_unknown)),
        !process_can_begin_create_v1(Seq::empty().push(pending)),
{
    let known_queues = Seq::empty().push(known);
    let unknown_queues = Seq::empty().push(unknown);
    let multiple_unknown_queues = Seq::empty().push(unknown).push(second_unknown);
    let pending_queues = Seq::empty().push(pending);
    assert(known_queue_id_reserved_v1(known_queues[0], queue_id));
    assert(unresolved_queue_identity_v1(unknown_queues[0]));
    assert(unresolved_queue_identity_v1(multiple_unknown_queues[0]));
    assert(unresolved_queue_identity_v1(multiple_unknown_queues[1]));
    assert(unresolved_queue_identity_v1(pending_queues[0]));
    assert(!returned_queue_id_available_v1(known_queues, queue_id)) by {
        if returned_queue_id_available_v1(known_queues, queue_id) {
            assert(!known_queue_id_reserved_v1(known_queues[0], queue_id));
        }
    }
    assert(!process_can_begin_create_v1(unknown_queues)) by {
        if process_can_begin_create_v1(unknown_queues) {
            assert(!unresolved_queue_identity_v1(unknown_queues[0]));
        }
    }
    assert(!process_can_begin_create_v1(multiple_unknown_queues)) by {
        if process_can_begin_create_v1(multiple_unknown_queues) {
            assert(!unresolved_queue_identity_v1(multiple_unknown_queues[0]));
        }
    }
    assert(at_most_one_create_pending_v1(multiple_unknown_queues));
    assert(!process_can_begin_create_v1(pending_queues)) by {
        if process_can_begin_create_v1(pending_queues) {
            assert(!unresolved_queue_identity_v1(pending_queues[0]));
        }
    }
}

pub proof fn history_append_preserves_exact_prefix_v1(
    history: Seq<QueueHistoryEntryV1>,
    entry: QueueHistoryEntryV1,
)
    requires
        entry.sequence == history.len() + 1,
    ensures
        append_history_v1(history, entry).len() == history.len() + 1,
        forall |i: int| 0 <= i < history.len()
            ==> #[trigger] append_history_v1(history, entry)[i] == history[i],
        append_history_v1(history, entry)[history.len() as int] == entry,
{
}

} // verus!
