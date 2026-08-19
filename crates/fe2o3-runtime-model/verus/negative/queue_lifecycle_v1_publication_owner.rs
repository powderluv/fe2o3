use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum PublicationOwnerV1 {
    Generic,
    ComputeAqlQueue { instance: nat, generation: nat },
}

pub struct PublicationLeaseV1 {
    pub owner: PublicationOwnerV1,
    pub live: bool,
}

pub open spec fn mutated_generic_release_ignores_owner_v1(
    lease: PublicationLeaseV1,
) -> Option<PublicationLeaseV1> {
    if lease.live {
        Some(PublicationLeaseV1 { owner: lease.owner, live: false })
    } else {
        None
    }
}

pub proof fn mutated_generic_release_rejects_queue_owner_v1(
    lease: PublicationLeaseV1,
)
    requires
        lease.live,
        lease.owner == (PublicationOwnerV1::ComputeAqlQueue {
            instance: 1,
            generation: 1,
        }),
    ensures
        mutated_generic_release_ignores_owner_v1(lease).is_none(),
{
}

} // verus!
