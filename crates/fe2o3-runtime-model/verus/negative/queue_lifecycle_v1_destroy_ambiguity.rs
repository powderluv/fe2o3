use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    DestroyPending,
    Destroyed,
    Ambiguous,
}

pub open spec fn queue_retains_resources_v1(phase: QueuePhaseV1) -> bool {
    phase != QueuePhaseV1::Destroyed
}

pub open spec fn mutated_indeterminate_destroy_v1() -> QueuePhaseV1 {
    QueuePhaseV1::Destroyed
}

pub proof fn mutated_indeterminate_destroy_remains_retaining_v1()
    ensures
        queue_retains_resources_v1(mutated_indeterminate_destroy_v1()),
{
}

} // verus!
