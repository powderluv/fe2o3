use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    Active,
    Ambiguous,
}

pub struct QueueRecordV1 {
    pub phase: QueuePhaseV1,
    pub queue_id: Option<nat>,
}

pub open spec fn mutated_id_reserved_only_by_active_v1(
    record: QueueRecordV1,
    queue_id: nat,
) -> bool {
    record.phase == QueuePhaseV1::Active && record.queue_id == Some(queue_id)
}

pub proof fn mutated_ambiguous_known_id_blocks_reuse_v1(
    record: QueueRecordV1,
    queue_id: nat,
)
    requires
        record.phase == QueuePhaseV1::Ambiguous,
        record.queue_id == Some(queue_id),
    ensures
        mutated_id_reserved_only_by_active_v1(record, queue_id),
{
}

} // verus!
