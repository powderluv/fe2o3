use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum QueueResourceRoleV1 {
    Ring,
    Control,
    EndOfPipe,
    ContextSave,
}

pub struct QueueResourceV1 {
    pub role: QueueResourceRoleV1,
    pub mapping_id: nat,
}

pub open spec fn mutated_substitute_eop_with_ring_v1(
    resources: Seq<QueueResourceV1>,
) -> Seq<QueueResourceV1> {
    resources.update(2, resources[0])
}

pub proof fn mutated_queue_resource_substitution_preserves_roles_v1(
    resources: Seq<QueueResourceV1>,
)
    requires
        resources.len() == 4,
        resources[0].role == QueueResourceRoleV1::Ring,
        resources[2].role == QueueResourceRoleV1::EndOfPipe,
    ensures
        mutated_substitute_eop_with_ring_v1(resources)[2].role
            == QueueResourceRoleV1::EndOfPipe,
{
}

} // verus!
