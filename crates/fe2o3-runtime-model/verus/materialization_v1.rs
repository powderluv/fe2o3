use vstd::prelude::*;

verus! {

#[derive(PartialEq, Eq)]
pub enum MaterializationActionV1 {
    ZeroFullImage,
    CopyFirst,
    CopySecond,
    CopyThird,
}

pub struct MaterializationSegmentV1 {
    pub source_start: nat,
    pub source_size: nat,
    pub memory_start: nat,
    pub memory_size: nat,
    pub mapping_start: nat,
    pub mapping_size: nat,
}

pub struct MaterializationPlanV1 {
    pub first: MaterializationSegmentV1,
    pub second: MaterializationSegmentV1,
    pub third: MaterializationSegmentV1,
    pub input_len: nat,
    pub image_start: nat,
    pub image_end: nat,
}

pub open spec fn max_u64_v1() -> nat {
    0xffffffffffffffff
}

pub open spec fn max_input_bytes_v1() -> nat {
    64 * 1024 * 1024
}

pub open spec fn max_image_span_v1() -> nat {
    64 * 1024 * 1024
}

pub open spec fn checked_u64_range_v1(start: nat, size: nat) -> bool {
    start <= max_u64_v1() && size <= max_u64_v1() - start
}

pub open spec fn checked_bounded_range_v1(
    bound: nat,
    start: nat,
    size: nat,
) -> bool {
    start <= bound && size <= bound - start
}

pub open spec fn segment_checked_v1(
    plan: MaterializationPlanV1,
    segment: MaterializationSegmentV1,
) -> bool {
    &&& segment.source_size > 0
    &&& segment.memory_size > 0
    &&& segment.source_size <= segment.memory_size
    &&& checked_u64_range_v1(segment.source_start, segment.source_size)
    &&& checked_u64_range_v1(segment.memory_start, segment.memory_size)
    &&& checked_u64_range_v1(segment.mapping_start, segment.mapping_size)
    &&& checked_bounded_range_v1(
        plan.input_len,
        segment.source_start,
        segment.source_size,
    )
    &&& plan.image_start <= segment.mapping_start
    &&& segment.mapping_start <= segment.memory_start
    &&& segment.memory_start + segment.memory_size
        <= segment.mapping_start + segment.mapping_size
    &&& segment.mapping_start + segment.mapping_size <= plan.image_end
}

pub open spec fn ranges_disjoint_v1(
    left_start: nat,
    left_size: nat,
    right_start: nat,
    right_size: nat,
) -> bool {
    left_start + left_size <= right_start
        || right_start + right_size <= left_start
}

pub open spec fn canonical_materialization_plan_v1(
    plan: MaterializationPlanV1,
) -> bool {
    &&& plan.input_len <= max_input_bytes_v1()
    &&& plan.image_start <= plan.image_end <= max_u64_v1()
    &&& plan.image_end - plan.image_start <= max_image_span_v1()
    &&& segment_checked_v1(plan, plan.first)
    &&& segment_checked_v1(plan, plan.second)
    &&& segment_checked_v1(plan, plan.third)
    &&& plan.first.mapping_start == plan.image_start
    &&& plan.first.mapping_start + plan.first.mapping_size
        <= plan.second.mapping_start
    &&& plan.second.mapping_start + plan.second.mapping_size
        <= plan.third.mapping_start
    &&& plan.third.mapping_start + plan.third.mapping_size == plan.image_end
    &&& ranges_disjoint_v1(
        plan.first.source_start,
        plan.first.source_size,
        plan.second.source_start,
        plan.second.source_size,
    )
    &&& ranges_disjoint_v1(
        plan.first.source_start,
        plan.first.source_size,
        plan.third.source_start,
        plan.third.source_size,
    )
    &&& ranges_disjoint_v1(
        plan.second.source_start,
        plan.second.source_size,
        plan.third.source_start,
        plan.third.source_size,
    )
}

pub open spec fn image_len_v1(plan: MaterializationPlanV1) -> nat {
    (plan.image_end - plan.image_start) as nat
}

pub open spec fn destination_start_v1(
    plan: MaterializationPlanV1,
    segment: MaterializationSegmentV1,
) -> nat {
    (segment.memory_start - plan.image_start) as nat
}

pub open spec fn mapping_start_v1(
    plan: MaterializationPlanV1,
    segment: MaterializationSegmentV1,
) -> nat {
    (segment.mapping_start - plan.image_start) as nat
}

pub open spec fn index_in_range_v1(index: int, start: nat, size: nat) -> bool {
    start as int <= index && index < (start + size) as int
}

pub open spec fn materialization_trace_v1() -> Seq<MaterializationActionV1> {
    seq![
        MaterializationActionV1::ZeroFullImage,
        MaterializationActionV1::CopyFirst,
        MaterializationActionV1::CopySecond,
        MaterializationActionV1::CopyThird,
    ]
}

pub open spec fn zero_image_v1(image_len: nat) -> Seq<u8> {
    Seq::new(image_len, |index: int| 0u8)
}

pub open spec fn copy_range_v1(
    input: Seq<u8>,
    before: Seq<u8>,
    source_start: nat,
    destination_start: nat,
    size: nat,
) -> Seq<u8> {
    Seq::new(before.len(), |index: int|
        if index_in_range_v1(index, destination_start, size) {
            input[source_start as int + index - destination_start as int]
        } else {
            before[index]
        }
    )
}

pub open spec fn zero_state_v1(plan: MaterializationPlanV1) -> Seq<u8> {
    zero_image_v1(image_len_v1(plan))
}

pub open spec fn after_first_copy_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
) -> Seq<u8> {
    copy_range_v1(
        input,
        zero_state_v1(plan),
        plan.first.source_start,
        destination_start_v1(plan, plan.first),
        plan.first.source_size,
    )
}

pub open spec fn after_second_copy_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
) -> Seq<u8> {
    copy_range_v1(
        input,
        after_first_copy_v1(plan, input),
        plan.second.source_start,
        destination_start_v1(plan, plan.second),
        plan.second.source_size,
    )
}

pub open spec fn materialized_image_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
) -> Seq<u8> {
    copy_range_v1(
        input,
        after_second_copy_v1(plan, input),
        plan.third.source_start,
        destination_start_v1(plan, plan.third),
        plan.third.source_size,
    )
}

pub open spec fn nonvacuity_plan_v1() -> MaterializationPlanV1 {
    MaterializationPlanV1 {
        first: MaterializationSegmentV1 {
            source_start: 0,
            source_size: 1,
            memory_start: 1,
            memory_size: 2,
            mapping_start: 0,
            mapping_size: 4,
        },
        second: MaterializationSegmentV1 {
            source_start: 1,
            source_size: 1,
            memory_start: 9,
            memory_size: 2,
            mapping_start: 8,
            mapping_size: 4,
        },
        third: MaterializationSegmentV1 {
            source_start: 2,
            source_size: 1,
            memory_start: 17,
            memory_size: 2,
            mapping_start: 16,
            mapping_size: 4,
        },
        input_len: 3,
        image_start: 0,
        image_end: 20,
    }
}

pub open spec fn nonvacuity_input_v1() -> Seq<u8> {
    seq![5u8, 6u8, 7u8]
}

pub open spec fn zero_range_v1(
    image: Seq<u8>,
    start: nat,
    size: nat,
) -> bool {
    &&& checked_bounded_range_v1(image.len(), start, size)
    &&& forall|offset: nat| offset < size ==>
        #[trigger] image[(start + offset) as int] == 0
}

pub open spec fn exact_copy_range_v1(
    input: Seq<u8>,
    image: Seq<u8>,
    source_start: nat,
    destination_start: nat,
    size: nat,
) -> bool {
    &&& checked_bounded_range_v1(input.len(), source_start, size)
    &&& checked_bounded_range_v1(image.len(), destination_start, size)
    &&& forall|offset: nat| offset < size ==>
        #[trigger] image[(destination_start + offset) as int]
            == input[(source_start + offset) as int]
}

pub open spec fn segment_zero_preservation_v1(
    plan: MaterializationPlanV1,
    segment: MaterializationSegmentV1,
    image: Seq<u8>,
) -> bool {
    &&& zero_range_v1(
        image,
        mapping_start_v1(plan, segment),
        (segment.memory_start - segment.mapping_start) as nat,
    )
    &&& zero_range_v1(
        image,
        destination_start_v1(plan, segment) + segment.source_size,
        (segment.memory_size - segment.source_size) as nat,
    )
    &&& zero_range_v1(
        image,
        destination_start_v1(plan, segment) + segment.memory_size,
        (segment.mapping_start + segment.mapping_size
            - (segment.memory_start + segment.memory_size)) as nat,
    )
}

pub open spec fn gaps_remain_zero_v1(
    plan: MaterializationPlanV1,
    image: Seq<u8>,
) -> bool {
    &&& zero_range_v1(
        image,
        mapping_start_v1(plan, plan.first) + plan.first.mapping_size,
        (plan.second.mapping_start
            - (plan.first.mapping_start + plan.first.mapping_size)) as nat,
    )
    &&& zero_range_v1(
        image,
        mapping_start_v1(plan, plan.second) + plan.second.mapping_size,
        (plan.third.mapping_start
            - (plan.second.mapping_start + plan.second.mapping_size)) as nat,
    )
}

pub proof fn full_zero_transition_initializes_every_byte_v1(image_len: nat)
    ensures
        zero_image_v1(image_len).len() == image_len,
        forall|index: int| 0 <= index < image_len ==>
            #[trigger] zero_image_v1(image_len)[index] == 0,
{
}

pub proof fn checked_copy_transition_is_exact_and_framed_v1(
    input: Seq<u8>,
    before: Seq<u8>,
    source_start: nat,
    destination_start: nat,
    size: nat,
)
    requires
        checked_bounded_range_v1(input.len(), source_start, size),
        checked_bounded_range_v1(before.len(), destination_start, size),
    ensures
        copy_range_v1(input, before, source_start, destination_start, size).len()
            == before.len(),
        forall|index: int| 0 <= index < before.len() ==>
            if index_in_range_v1(index, destination_start, size) {
                #[trigger] copy_range_v1(
                    input, before, source_start, destination_start, size,
                )[index] == input[source_start as int + index - destination_start as int]
            } else {
                #[trigger] copy_range_v1(
                    input, before, source_start, destination_start, size,
                )[index] == before[index]
            },
{
}

pub proof fn canonical_materialization_ranges_are_checked_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
)
    requires
        canonical_materialization_plan_v1(plan),
        input.len() == plan.input_len,
    ensures
        input.len() == plan.input_len <= max_input_bytes_v1(),
        image_len_v1(plan) <= max_image_span_v1(),
        checked_bounded_range_v1(
            input.len(), plan.first.source_start, plan.first.source_size,
        ),
        checked_bounded_range_v1(
            input.len(), plan.second.source_start, plan.second.source_size,
        ),
        checked_bounded_range_v1(
            input.len(), plan.third.source_start, plan.third.source_size,
        ),
        checked_bounded_range_v1(
            image_len_v1(plan), destination_start_v1(plan, plan.first),
            plan.first.source_size,
        ),
        checked_bounded_range_v1(
            image_len_v1(plan), destination_start_v1(plan, plan.second),
            plan.second.source_size,
        ),
        checked_bounded_range_v1(
            image_len_v1(plan), destination_start_v1(plan, plan.third),
            plan.third.source_size,
        ),
        ranges_disjoint_v1(
            destination_start_v1(plan, plan.first), plan.first.source_size,
            destination_start_v1(plan, plan.second), plan.second.source_size,
        ),
        ranges_disjoint_v1(
            destination_start_v1(plan, plan.first), plan.first.source_size,
            destination_start_v1(plan, plan.third), plan.third.source_size,
        ),
        ranges_disjoint_v1(
            destination_start_v1(plan, plan.second), plan.second.source_size,
            destination_start_v1(plan, plan.third), plan.third.source_size,
        ),
{
}

pub proof fn canonical_zero_then_copy_execution_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
)
    requires
        canonical_materialization_plan_v1(plan),
        input.len() == plan.input_len,
    ensures
        materialization_trace_v1().len() == 4,
        materialization_trace_v1()[0] == MaterializationActionV1::ZeroFullImage,
        materialization_trace_v1()[1] == MaterializationActionV1::CopyFirst,
        materialization_trace_v1()[2] == MaterializationActionV1::CopySecond,
        materialization_trace_v1()[3] == MaterializationActionV1::CopyThird,
        zero_state_v1(plan).len() == image_len_v1(plan),
        after_first_copy_v1(plan, input).len() == image_len_v1(plan),
        after_second_copy_v1(plan, input).len() == image_len_v1(plan),
        materialized_image_v1(plan, input).len() == image_len_v1(plan),
        forall|index: int| 0 <= index < image_len_v1(plan) ==>
            #[trigger] materialized_image_v1(plan, input)[index]
                == if index_in_range_v1(
                    index, destination_start_v1(plan, plan.third),
                    plan.third.source_size,
                ) {
                    input[plan.third.source_start as int + index
                        - destination_start_v1(plan, plan.third) as int]
                } else if index_in_range_v1(
                    index, destination_start_v1(plan, plan.second),
                    plan.second.source_size,
                ) {
                    input[plan.second.source_start as int + index
                        - destination_start_v1(plan, plan.second) as int]
                } else if index_in_range_v1(
                    index, destination_start_v1(plan, plan.first),
                    plan.first.source_size,
                ) {
                    input[plan.first.source_start as int + index
                        - destination_start_v1(plan, plan.first) as int]
                } else {
                    0u8
                },
{
    canonical_materialization_ranges_are_checked_v1(plan, input);
    full_zero_transition_initializes_every_byte_v1(image_len_v1(plan));
    checked_copy_transition_is_exact_and_framed_v1(
        input,
        zero_state_v1(plan),
        plan.first.source_start,
        destination_start_v1(plan, plan.first),
        plan.first.source_size,
    );
    checked_copy_transition_is_exact_and_framed_v1(
        input,
        after_first_copy_v1(plan, input),
        plan.second.source_start,
        destination_start_v1(plan, plan.second),
        plan.second.source_size,
    );
    checked_copy_transition_is_exact_and_framed_v1(
        input,
        after_second_copy_v1(plan, input),
        plan.third.source_start,
        destination_start_v1(plan, plan.third),
        plan.third.source_size,
    );
}

pub proof fn canonical_materialization_copies_exact_sources_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
)
    requires
        canonical_materialization_plan_v1(plan),
        input.len() == plan.input_len,
    ensures
        exact_copy_range_v1(
            input, materialized_image_v1(plan, input), plan.first.source_start,
            destination_start_v1(plan, plan.first), plan.first.source_size,
        ),
        exact_copy_range_v1(
            input, materialized_image_v1(plan, input), plan.second.source_start,
            destination_start_v1(plan, plan.second), plan.second.source_size,
        ),
        exact_copy_range_v1(
            input, materialized_image_v1(plan, input), plan.third.source_start,
            destination_start_v1(plan, plan.third), plan.third.source_size,
        ),
{
    canonical_materialization_ranges_are_checked_v1(plan, input);
    canonical_zero_then_copy_execution_v1(plan, input);
}

pub proof fn materialized_range_outside_copies_is_zero_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
    start: nat,
    size: nat,
)
    requires
        canonical_materialization_plan_v1(plan),
        input.len() == plan.input_len,
        checked_bounded_range_v1(image_len_v1(plan), start, size),
        ranges_disjoint_v1(
            start, size, destination_start_v1(plan, plan.first),
            plan.first.source_size,
        ),
        ranges_disjoint_v1(
            start, size, destination_start_v1(plan, plan.second),
            plan.second.source_size,
        ),
        ranges_disjoint_v1(
            start, size, destination_start_v1(plan, plan.third),
            plan.third.source_size,
        ),
    ensures
        zero_range_v1(materialized_image_v1(plan, input), start, size),
{
    canonical_zero_then_copy_execution_v1(plan, input);
}

pub proof fn canonical_materialization_preserves_all_uncopied_regions_v1(
    plan: MaterializationPlanV1,
    input: Seq<u8>,
)
    requires
        canonical_materialization_plan_v1(plan),
        input.len() == plan.input_len,
    ensures
        segment_zero_preservation_v1(
            plan, plan.first, materialized_image_v1(plan, input),
        ),
        segment_zero_preservation_v1(
            plan, plan.second, materialized_image_v1(plan, input),
        ),
        segment_zero_preservation_v1(
            plan, plan.third, materialized_image_v1(plan, input),
        ),
        gaps_remain_zero_v1(plan, materialized_image_v1(plan, input)),
{
    canonical_materialization_ranges_are_checked_v1(plan, input);
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        mapping_start_v1(plan, plan.first),
        (plan.first.memory_start - plan.first.mapping_start) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        destination_start_v1(plan, plan.first) + plan.first.source_size,
        (plan.first.memory_size - plan.first.source_size) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        destination_start_v1(plan, plan.first) + plan.first.memory_size,
        (plan.first.mapping_start + plan.first.mapping_size
            - (plan.first.memory_start + plan.first.memory_size)) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        mapping_start_v1(plan, plan.second),
        (plan.second.memory_start - plan.second.mapping_start) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        destination_start_v1(plan, plan.second) + plan.second.source_size,
        (plan.second.memory_size - plan.second.source_size) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        destination_start_v1(plan, plan.second) + plan.second.memory_size,
        (plan.second.mapping_start + plan.second.mapping_size
            - (plan.second.memory_start + plan.second.memory_size)) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        mapping_start_v1(plan, plan.third),
        (plan.third.memory_start - plan.third.mapping_start) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        destination_start_v1(plan, plan.third) + plan.third.source_size,
        (plan.third.memory_size - plan.third.source_size) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        destination_start_v1(plan, plan.third) + plan.third.memory_size,
        (plan.third.mapping_start + plan.third.mapping_size
            - (plan.third.memory_start + plan.third.memory_size)) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        mapping_start_v1(plan, plan.first) + plan.first.mapping_size,
        (plan.second.mapping_start
            - (plan.first.mapping_start + plan.first.mapping_size)) as nat,
    );
    materialized_range_outside_copies_is_zero_v1(
        plan,
        input,
        mapping_start_v1(plan, plan.second) + plan.second.mapping_size,
        (plan.third.mapping_start
            - (plan.second.mapping_start + plan.second.mapping_size)) as nat,
    );
}

pub proof fn canonical_materialization_nonvacuity_witness_v1()
    ensures
        canonical_materialization_plan_v1(nonvacuity_plan_v1()),
        nonvacuity_input_v1().len() == nonvacuity_plan_v1().input_len,
        materialized_image_v1(
            nonvacuity_plan_v1(), nonvacuity_input_v1(),
        ).len() == nonvacuity_plan_v1().image_end - nonvacuity_plan_v1().image_start,
        materialized_image_v1(
            nonvacuity_plan_v1(), nonvacuity_input_v1(),
        )[1] == 5,
        materialized_image_v1(
            nonvacuity_plan_v1(), nonvacuity_input_v1(),
        )[0] == 0,
{
    assert(canonical_materialization_plan_v1(nonvacuity_plan_v1()));
    canonical_zero_then_copy_execution_v1(nonvacuity_plan_v1(), nonvacuity_input_v1());
}

} // verus!
