#[derive(Clone, Copy)]
enum SubgroupForwardingTargetV1 {
    Terminal(Value),
    Phi(usize),
}

#[derive(Clone, Copy)]
enum SubgroupForwardingStateV1 {
    Unvisited,
    Active(usize),
    Resolved(Value),
}

struct SubgroupForwardingRowV1<'a> {
    value: Value,
    inputs: &'a [Value],
    bucket_next: Option<usize>,
    target: SubgroupForwardingTargetV1,
    state: SubgroupForwardingStateV1,
}

struct SubgroupForwardingIndexV1<'a> {
    entry: Ptr<BasicBlock>,
    capacity: usize,
    comparison_bound: usize,
    buckets: BTreeMap<u64, usize>,
    rows: Vec<SubgroupForwardingRowV1<'a>>,
    path: Vec<usize>,
}

struct ResolvedSubgroupForwardingV1<'a>(SubgroupForwardingIndexV1<'a>);

fn forwarding_error_v1(detail: &'static str) -> PlironTensorLayoutFindingV1 {
    PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
        detail: detail.to_owned(),
    }
}

fn subgroup_forwarding_storage_items_v1(capacity: usize) -> Option<usize> {
    if capacity == 0 {
        return Some(0);
    }
    // Requested row/path words, plus a conservative entire pinned B-tree
    // node per distinct hash. This is not allocator-byte-exact accounting.
    const WORD: usize = std::mem::size_of::<usize>();
    const ROW: usize = std::mem::size_of::<SubgroupForwardingRowV1<'static>>().div_ceil(WORD);
    const HEADERS: usize = std::mem::size_of::<SubgroupForwardingIndexV1<'static>>().div_ceil(WORD);
    capacity.checked_mul(ROW + 1 + 64)?.checked_add(HEADERS)
}

fn subgroup_forwarding_hash_v1(
    value: Value,
    work: &mut usize,
) -> Result<u64, PlironTensorLayoutFindingV1> {
    use std::hash::{Hash as _, Hasher as _};

    // The pinned Value hash contains only its fixed-size UID, entity tag,
    // owner, and arena key. Cover input traversal and fixed hasher setup/finalization.
    charge_uniformity_collection(work, 16 + 2 * std::mem::size_of::<Value>())?;
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hash);
    Ok(hash.finish())
}

fn subgroup_forwarding_type_v1(
    context: &Context,
    value: Value,
    work: &mut usize,
) -> Result<pliron::r#type::TypeHandle, PlironTensorLayoutFindingV1> {
    use pliron::r#type::Typed as _;

    charge_uniformity_collection(work, 2)?;
    let width = match value.defining_entity() {
        pliron::value::DefiningEntity::Op(operation) => operation
            .try_deref(context)
            .map_err(|_| forwarding_error_v1("forwarded value has a foreign or missing owner"))?
            .get_num_results(),
        pliron::value::DefiningEntity::Block(block) => block
            .try_deref(context)
            .map_err(|_| forwarding_error_v1("forwarded value has a foreign or missing owner"))?
            .get_num_arguments(),
    };
    // Both the validity check and Typed::get_type scan the defining roster.
    charge_uniformity_collection(
        work,
        width
            .checked_mul(2)
            .and_then(|width| width.checked_add(3))
            .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?,
    )?;
    value
        .try_find_index(context)
        .map_err(|_| forwarding_error_v1("forwarded value is absent from its defining roster"))?;
    Ok(value.get_type(context))
}

impl<'a> SubgroupForwardingIndexV1<'a> {
    fn new(
        entry: Ptr<BasicBlock>,
        capacity: usize,
        work: &mut usize,
    ) -> Result<Self, PlironTensorLayoutFindingV1> {
        ensure_uniformity_value_capacity(0, capacity)?;
        let storage = subgroup_forwarding_storage_items_v1(capacity)
            .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
        // Separately pay aggregate B-tree insertion movement: at most ten
        // shifted key/value pairs per insert, plus amortized split/parent moves.
        // With pinned B=6 and no removals, 64 units per admitted key covers both;
        // eight more cover the eventual key/node release traversal.
        let construction = capacity
            .checked_mul(72)
            .and_then(|movement| storage.checked_add(movement))
            .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
        charge_uniformity_collection(work, construction)?;
        let height = usize::BITS as usize
            - capacity
                .checked_add(1)
                .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?
                .leading_zeros() as usize;
        Ok(Self {
            entry,
            capacity,
            comparison_bound: height * 8,
            buckets: BTreeMap::new(),
            rows: uniformity_vec_v1(capacity)?,
            path: uniformity_vec_v1(capacity)?,
        })
    }

    fn lookup_hashed(
        &self,
        value: Value,
        hash: u64,
        work: &mut usize,
    ) -> Result<Option<usize>, PlironTensorLayoutFindingV1> {
        charge_uniformity_collection(work, self.comparison_bound + 1)?;
        let mut row = self.buckets.get(&hash).copied();
        while let Some(index) = row {
            charge_uniformity_collection(work, std::mem::size_of::<Value>() + 2)?;
            let candidate = &self.rows[index];
            if candidate.value == value {
                return Ok(Some(index));
            }
            row = candidate.bucket_next;
        }
        Ok(None)
    }

    fn lookup(
        &self,
        value: Value,
        work: &mut usize,
    ) -> Result<Option<usize>, PlironTensorLayoutFindingV1> {
        let hash = subgroup_forwarding_hash_v1(value, work)?;
        self.lookup_hashed(value, hash, work)
    }

    fn push_hashed(
        &mut self,
        value: Value,
        inputs: &'a [Value],
        hash: u64,
        work: &mut usize,
    ) -> Result<(), PlironTensorLayoutFindingV1> {
        charge_uniformity_collection(work, 2)?;
        if self.rows.len() == self.capacity {
            return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
        }
        if value
            .defining_block()
            .is_none_or(|block| block == self.entry)
        {
            return Err(forwarding_error_v1(
                "forwarding identity requires a non-entry block argument",
            ));
        }
        if self.lookup_hashed(value, hash, work)?.is_some() {
            return Err(forwarding_error_v1(
                "forwarding identity has a duplicate block argument",
            ));
        }
        // Every possible tree node and row was admitted by new before mutation.
        charge_uniformity_collection(work, self.comparison_bound + 3)?;
        let bucket_next = self.buckets.insert(hash, self.rows.len());
        self.rows.push(SubgroupForwardingRowV1 {
            value,
            inputs,
            bucket_next,
            target: SubgroupForwardingTargetV1::Terminal(value),
            state: SubgroupForwardingStateV1::Unvisited,
        });
        Ok(())
    }

    fn push(
        &mut self,
        value: Value,
        inputs: &'a [Value],
        work: &mut usize,
    ) -> Result<(), PlironTensorLayoutFindingV1> {
        let hash = subgroup_forwarding_hash_v1(value, work)?;
        self.push_hashed(value, inputs, hash, work)
    }

    fn finish(
        mut self,
        context: &Context,
        work: &mut usize,
    ) -> Result<ResolvedSubgroupForwardingV1<'a>, PlironTensorLayoutFindingV1> {
        charge_uniformity_collection(work, 1)?;
        if self.rows.len() != self.capacity {
            return Err(forwarding_error_v1(
                "forwarding identity has an incomplete phi roster",
            ));
        }
        for index in 0..self.rows.len() {
            charge_uniformity_collection(work, 1)?;
            let row = &self.rows[index];
            let argument_type = subgroup_forwarding_type_v1(context, row.value, work)?;
            let Some(&first) = row.inputs.first() else {
                continue;
            };
            charge_uniformity_collection(
                work,
                row.inputs
                    .len()
                    .checked_mul(std::mem::size_of::<Value>())
                    .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?,
            )?;
            if row.inputs.iter().any(|input| *input != first) {
                continue;
            }
            let incoming_type = subgroup_forwarding_type_v1(context, first, work)?;
            charge_uniformity_collection(work, 2)?;
            if argument_type != incoming_type {
                return Err(forwarding_error_v1(
                    "forwarding identity has unequal edge types",
                ));
            }
            let target = match self.lookup(first, work)? {
                Some(target) => SubgroupForwardingTargetV1::Phi(target),
                None => SubgroupForwardingTargetV1::Terminal(first),
            };
            charge_uniformity_collection(work, 1)?;
            self.rows[index].target = target;
        }
        self.resolve(work)?.refine(context, work)
    }

    fn resolve(
        mut self,
        work: &mut usize,
    ) -> Result<ResolvedSubgroupForwardingV1<'a>, PlironTensorLayoutFindingV1> {
        for start in 0..self.rows.len() {
            charge_uniformity_collection(work, 1)?;
            if matches!(
                self.rows[start].state,
                SubgroupForwardingStateV1::Resolved(_)
            ) {
                continue;
            }
            let mut current = start;
            let representative = loop {
                charge_uniformity_collection(work, 3)?;
                match self.rows[current].state {
                    SubgroupForwardingStateV1::Resolved(value) => break value,
                    SubgroupForwardingStateV1::Active(offset) => {
                        let cycle = &self.path[offset..];
                        charge_uniformity_collection(work, cycle.len() + 1)?;
                        let representative = *cycle.iter().min().ok_or_else(|| {
                            forwarding_error_v1("forwarding identity has an empty active cycle")
                        })?;
                        break self.rows[representative].value;
                    }
                    SubgroupForwardingStateV1::Unvisited => {
                        charge_uniformity_collection(work, 2)?;
                        self.rows[current].state =
                            SubgroupForwardingStateV1::Active(self.path.len());
                        self.path.push(current);
                        match self.rows[current].target {
                            SubgroupForwardingTargetV1::Terminal(value) => break value,
                            SubgroupForwardingTargetV1::Phi(next) => current = next,
                        }
                    }
                }
            };
            while !self.path.is_empty() {
                charge_uniformity_collection(work, 3)?;
                let index = self.path.pop().ok_or_else(|| {
                    forwarding_error_v1("forwarding identity lost its pending path")
                })?;
                self.rows[index].state = SubgroupForwardingStateV1::Resolved(representative);
            }
        }
        Ok(ResolvedSubgroupForwardingV1(self))
    }
}

impl ResolvedSubgroupForwardingV1<'_> {
    fn refine(
        mut self,
        context: &Context,
        work: &mut usize,
    ) -> Result<Self, PlironTensorLayoutFindingV1> {
        if self.0.rows.is_empty() {
            return Ok(self);
        }
        loop {
            let mut changed = false;
            for index in 0..self.0.rows.len() {
                charge_uniformity_collection(work, std::mem::size_of::<Value>() + 2)?;
                let value = self.0.rows[index].value;
                if !matches!(self.0.rows[index].state, SubgroupForwardingStateV1::Resolved(root) if root == value)
                {
                    continue;
                }
                let mut common = None;
                let mut different = false;
                for input in self.0.rows[index].inputs {
                    let next = self.representative(*input, work)?;
                    charge_uniformity_collection(work, 2 * std::mem::size_of::<Value>() + 2)?;
                    // phi(a, self) forwards a; a self-only cycle has no anchor.
                    if next == value {
                        continue;
                    }
                    if common.is_some_and(|previous| previous != next) {
                        different = true;
                        break;
                    }
                    common = Some(next);
                }
                let Some(common) = common.filter(|_| !different) else {
                    continue;
                };
                let expected = subgroup_forwarding_type_v1(context, value, work)?;
                for input in self.0.rows[index].inputs {
                    let actual = subgroup_forwarding_type_v1(context, *input, work)?;
                    charge_uniformity_collection(work, 1)?;
                    if actual != expected {
                        return Err(forwarding_error_v1(
                            "forwarding identity has unequal edge types",
                        ));
                    }
                }
                charge_uniformity_collection(work, 1)?;
                self.0.rows[index].state = SubgroupForwardingStateV1::Resolved(common);
                changed = true;
            }
            if !changed {
                // Flatten the refined chains before the immutable query phase.
                for index in 0..self.0.rows.len() {
                    charge_uniformity_collection(work, 1)?;
                    let root = self.representative(self.0.rows[index].value, work)?;
                    charge_uniformity_collection(work, 1)?;
                    self.0.rows[index].state = SubgroupForwardingStateV1::Resolved(root);
                }
                return Ok(self);
            }
        }
    }
    fn representative(
        &self,
        value: Value,
        work: &mut usize,
    ) -> Result<Value, PlironTensorLayoutFindingV1> {
        let mut value = value;
        for _ in 0..=self.0.rows.len() {
            let Some(index) = self.0.lookup(value, work)? else {
                return Ok(value);
            };
            charge_uniformity_collection(work, std::mem::size_of::<Value>() + 1)?;
            match self.0.rows[index].state {
                SubgroupForwardingStateV1::Resolved(next) => {
                    if next == value {
                        return Ok(value);
                    }
                    value = next;
                }
                _ => {
                    return Err(forwarding_error_v1(
                        "forwarding identity has an unresolved representative",
                    ));
                }
            }
        }
        Err(forwarding_error_v1(
            "forwarding identity has a cyclic refined representative",
        ))
    }

    fn inputs_equal(
        &self,
        index: usize,
        work: &mut usize,
    ) -> Result<bool, PlironTensorLayoutFindingV1> {
        charge_uniformity_collection(work, 1)?;
        let inputs = self.0.rows[index].inputs;
        if inputs.len() < 2 {
            return Ok(true);
        }
        let first = inputs[0];
        let first = self.representative(first, work)?;
        for input in &inputs[1..] {
            charge_uniformity_collection(work, 1)?;
            let next = self.representative(*input, work)?;
            charge_uniformity_collection(work, std::mem::size_of::<Value>())?;
            if next != first {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "forwarded_phi_equality_tests.rs"]
mod forwarded_phi_equality_tests;
