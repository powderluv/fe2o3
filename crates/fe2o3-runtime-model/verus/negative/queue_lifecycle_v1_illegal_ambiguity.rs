use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    Active,
    UpdatePending,
    Ambiguous,
}

pub open spec fn mutated_indeterminate_update_ignores_phase_v1(
    phase: QueuePhaseV1,
) -> QueuePhaseV1 {
    QueuePhaseV1::Ambiguous
}

pub proof fn mutated_illegal_indeterminate_update_preserves_active_v1()
    ensures
        mutated_indeterminate_update_ignores_phase_v1(QueuePhaseV1::Active)
            == QueuePhaseV1::Active,
{
}

} // verus!
