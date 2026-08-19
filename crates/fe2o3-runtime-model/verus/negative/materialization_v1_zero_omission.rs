use vstd::prelude::*;

verus! {

pub open spec fn mutated_zero_image_v1(image_len: nat) -> Seq<u8> {
    Seq::new(image_len, |index: int|
        if index == 0 {
            1u8
        } else {
            0u8
        }
    )
}

pub proof fn mutated_zero_first_initializes_every_byte_v1()
    ensures
        mutated_zero_image_v1(2).len() == 2,
        forall|index: int| 0 <= index < 2 ==>
            #[trigger] mutated_zero_image_v1(2)[index] == 0,
{
}

} // verus!
