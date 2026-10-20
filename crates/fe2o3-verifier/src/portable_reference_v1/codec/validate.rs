use super::*;

pub(super) fn input(
    input: &NativeCpuInputV1<'_>,
    bytes: usize,
    s: &mut Scope<'_, '_>,
) -> Result<(), Error> {
    subject(&input.subject(), bytes, s)
}

pub(super) fn subject(
    input: &CpuSubject<'_>,
    bytes: usize,
    s: &mut Scope<'_, '_>,
) -> Result<(), Error> {
    let signature = input.replay.signature_preimage;
    s.work(mul(
        8,
        add(
            signature.kernel_inputs().len(),
            signature.reference_inputs().len(),
        )?,
    )?)?;
    let derived = signature.derive_relations_v1()?;
    let ir = input.replay.effect_ir;
    require(
        derived.len() == ir.relations.len() && derived.len() == ir.argument_count as usize,
        "signature relation arity",
    )?;
    for (raw, relation) in ir.relations.iter().enumerate() {
        s.work(8)?;
        require(
            derived.relation_at_raw_argument_v1(raw as u32) == Some(*relation),
            "signature relation",
        )?;
    }
    // The complete bounded grammar walk precedes this legacy hash traversal.
    s.work(mul(4, bytes)?)?;
    require(
        ir.canonical_sha256_v1() == input.replay.effect_ir_sha256,
        "legacy effect digest",
    )?;
    acyclic(ir, s)
}

fn acyclic(ir: &ReferenceEffectIrV1, s: &mut Scope<'_, '_>) -> Result<(), Error> {
    let mut indegree = s.vector::<usize>(ir.blocks.len())?;
    for _ in &ir.blocks {
        indegree.push(0);
    }
    for block in &ir.blocks {
        for target in extraction::reference_successors_v1(&block.terminator) {
            s.work(1)?;
            let degree = indegree
                .get_mut(target as usize)
                .ok_or(Error::Wire("block edge"))?;
            *degree = add(*degree, 1)?;
        }
        if let ReferenceTerminatorV1::Switch { values, .. } = &block.terminator {
            let mut keys = s.vector(values.len())?;
            // Conservative comparison allowance; sorting scratch never changes
            // source-order cases in the retained body or canonical wire frame.
            s.work(mul(4, mul(add(values.len(), 1)?, add(values.len(), 1)?)?)?)?;
            for (key, _) in values {
                keys.push(*key);
            }
            keys.sort_unstable();
            require(
                keys.windows(2).all(|pair| pair[0] != pair[1]),
                "duplicate switch value",
            )?;
        }
    }
    let mut queue = s.vector(ir.blocks.len())?;
    for (index, degree) in indegree.iter().enumerate() {
        s.work(1)?;
        if *degree == 0 {
            queue.push(index);
        }
    }
    let mut cursor = 0;
    while cursor < queue.len() {
        s.work(1)?;
        let block = &ir.blocks[queue[cursor]];
        cursor += 1;
        for target in extraction::reference_successors_v1(&block.terminator) {
            s.work(1)?;
            let degree = &mut indegree[target as usize];
            *degree = degree.checked_sub(1).ok_or(Resource::Arithmetic)?;
            if *degree == 0 {
                require(queue.len() < queue.capacity(), "topological queue")?;
                queue.push(target as usize);
            }
        }
    }
    require(cursor == ir.blocks.len(), "cyclic CPU body")
}
