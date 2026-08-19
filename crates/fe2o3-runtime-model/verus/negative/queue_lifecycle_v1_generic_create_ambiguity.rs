use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueuePhaseV1 {
    CreatePending,
    UpdatePending,
}

#[derive(PartialEq, Eq)]
pub enum QueueOperationV1 {
    Create,
    Update,
}

pub open spec fn mutated_legal_non_create_pending_operation_v1(
    operation: QueueOperationV1,
    phase: QueuePhaseV1,
) -> bool {
    match operation {
        QueueOperationV1::Create => phase == QueuePhaseV1::CreatePending,
        QueueOperationV1::Update => phase == QueuePhaseV1::UpdatePending,
    }
}

pub proof fn mutated_generic_create_ambiguity_is_excluded_v1()
    ensures
        !mutated_legal_non_create_pending_operation_v1(
            QueueOperationV1::Create,
            QueuePhaseV1::CreatePending,
        ),
{
}

} // verus!
