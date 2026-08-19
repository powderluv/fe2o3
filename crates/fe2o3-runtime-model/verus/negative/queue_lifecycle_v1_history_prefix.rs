use vstd::prelude::*;

verus! {

pub open spec fn mutated_overwrite_history_head_v1(
    history: Seq<nat>,
    entry: nat,
) -> Seq<nat> {
    history.update(0, entry)
}

pub proof fn mutated_queue_history_overwrite_preserves_prefix_v1(
    history: Seq<nat>,
    entry: nat,
)
    requires
        history.len() > 0,
        entry != history[0],
    ensures
        forall |i: int| 0 <= i < history.len()
            ==> #[trigger] mutated_overwrite_history_head_v1(history, entry)[i] == history[i],
{
}

} // verus!
