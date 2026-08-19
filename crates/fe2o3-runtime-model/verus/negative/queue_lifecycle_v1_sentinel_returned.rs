use vstd::prelude::*;

verus! {

pub open spec fn max_u32_queue_id_v1() -> nat {
    0xffff_ffff
}

pub open spec fn mutated_classify_returned_queue_id_v1(queue_id: nat) -> Option<nat> {
    if queue_id <= max_u32_queue_id_v1() {
        Some(queue_id)
    } else {
        None
    }
}

pub proof fn mutated_returned_sentinel_is_rejected_v1()
    ensures
        mutated_classify_returned_queue_id_v1(max_u32_queue_id_v1()).is_none(),
{
}

} // verus!
