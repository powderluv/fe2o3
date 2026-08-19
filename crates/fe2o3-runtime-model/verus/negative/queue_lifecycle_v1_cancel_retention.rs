use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    CancelledBeforeCreate,
    Destroyed,
}

pub open spec fn mutated_only_destroy_stops_retention_v1(phase: QueuePhaseV1) -> bool {
    phase != QueuePhaseV1::Destroyed
}

pub proof fn mutated_cancelled_plan_is_nonretaining_v1()
    ensures
        !mutated_only_destroy_stops_retention_v1(QueuePhaseV1::CancelledBeforeCreate),
{
}

} // verus!
