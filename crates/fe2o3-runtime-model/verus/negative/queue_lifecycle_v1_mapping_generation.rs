use vstd::prelude::*;

verus! {

pub struct MappingKeyV1 {
    pub vm: nat,
    pub allocation_id: nat,
    pub allocation_generation: nat,
    pub mapping_id: nat,
}

pub open spec fn mutated_mapping_equal_ignores_generation_v1(
    left: MappingKeyV1,
    right: MappingKeyV1,
) -> bool {
    left.vm == right.vm
        && left.allocation_id == right.allocation_id
        && left.mapping_id == right.mapping_id
}

pub proof fn mutated_mapping_generation_substitution_is_exact_v1(
    left: MappingKeyV1,
    right: MappingKeyV1,
)
    requires
        left.vm == right.vm,
        left.allocation_id == right.allocation_id,
        left.mapping_id == right.mapping_id,
        left.allocation_generation != right.allocation_generation,
    ensures
        mutated_mapping_equal_ignores_generation_v1(left, right)
            ==> left.allocation_generation == right.allocation_generation,
{
}

} // verus!
