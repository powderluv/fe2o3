use vstd::prelude::*;

verus! {

pub open spec fn index_in_range_v1(index: int, start: nat, size: nat) -> bool {
    start as int <= index && index < (start + size) as int
}

pub open spec fn mutated_copy_range_v1(
    input: Seq<u8>,
    before: Seq<u8>,
    source_start: nat,
    destination_start: nat,
    size: nat,
) -> Seq<u8> {
    Seq::new(before.len(), |index: int|
        if index_in_range_v1(index, destination_start, size) {
            input[source_start as int + index - destination_start as int + 1]
        } else {
            before[index]
        }
    )
}

pub proof fn mutated_source_substitution_preserves_exact_byte_v1(
    input: Seq<u8>,
    before: Seq<u8>,
)
    requires
        input.len() == 2,
        before.len() == 1,
        input[0] == 7,
        input[1] == 9,
    ensures
        mutated_copy_range_v1(input, before, 0, 0, 1)[0] == input[0],
{
}

} // verus!
