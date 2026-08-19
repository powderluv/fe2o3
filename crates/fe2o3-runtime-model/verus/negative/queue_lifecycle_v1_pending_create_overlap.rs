use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    CreatePending,
    Ambiguous,
}

pub open spec fn mutated_unresolved_only_after_observation_v1(
    phase: QueuePhaseV1,
    queue_id_known: bool,
) -> bool {
    phase == QueuePhaseV1::Ambiguous && !queue_id_known
}

pub proof fn mutated_pending_create_blocks_second_begin_v1()
    ensures
        mutated_unresolved_only_after_observation_v1(
            QueuePhaseV1::CreatePending,
            false,
        ),
{
}

} // verus!
