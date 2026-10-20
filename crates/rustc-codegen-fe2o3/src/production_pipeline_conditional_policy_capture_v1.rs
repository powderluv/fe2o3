//! Inert snapshot of the policies borrowed during the original source visit.
use super::{Budget, Error, Resource};
use fe2o3_compiler_lineage::{
    MAX_NATIVE_CONDITIONAL_POLICY_ROSTER_ROOTS_V1,
    MAX_NATIVE_CONDITIONAL_POLICY_SIGNERS_PER_ROOT_V1,
    NATIVE_CONDITIONAL_POLICY_ROSTER_WORKING_STORAGE_V1, NativeConditionalPolicyRootInputV1,
    NativeConditionalPolicyRosterInputV1, NativeConditionalPolicyRosterLayoutV1,
    encode_native_conditional_policy_roster_v1,
};
use fe2o3_functional_proof::VerusToolchainIdentityV2;
use fe2o3_verifier::NativeConditionalRootPolicyV2;
use std::mem::size_of;

const SCRATCH: usize = NATIVE_CONDITIONAL_POLICY_ROSTER_WORKING_STORAGE_V1
    + size_of::<NativeConditionalPolicyRosterInputV1<'static>>()
    + size_of::<NativeConditionalPolicyRosterLayoutV1>()
    + size_of::<Error>()
    + 512;

pub(super) fn retained_storage(bytes: &Vec<u8>) -> Result<usize, Resource> {
    size_of::<Vec<u8>>()
        .checked_add(bytes.capacity())
        .ok_or(Resource::Arithmetic)
}

// No scratch refunds here: the enclosing source assembler releases them only
// after both original ledgers pass their postchecks. Failures remain terminal.
pub(super) fn capture(
    packet: &[u8],
    policies: &[NativeConditionalRootPolicyV2<'_>],
    roots: impl ExactSizeIterator<Item = (u32, [u8; 32])>,
    budget: &mut Budget<'_>,
) -> Result<Vec<u8>, Error> {
    budget.charge_work(1)?;
    if policies.is_empty()
        || policies.len() > MAX_NATIVE_CONDITIONAL_POLICY_ROSTER_ROOTS_V1
        || roots.len() != policies.len()
    {
        return Err(Error::Mismatch(
            "complete original conditional policy roster",
        ));
    }
    budget.reserve_storage(SCRATCH)?;
    let mut signer_count = 0usize;
    for policy in policies {
        budget.charge_work(1)?;
        let count = policy.effects.signer_identities().len();
        if count == 0 || count > MAX_NATIVE_CONDITIONAL_POLICY_SIGNERS_PER_ROOT_V1 {
            return Err(Error::Mismatch("bounded original effect signer set"));
        }
        signer_count = signer_count
            .checked_add(count)
            .ok_or(Resource::Arithmetic)?;
    }
    let mut signers = vector(signer_count, budget)?;
    for policy in policies {
        budget.charge_work(
            policy
                .effects
                .signer_identities()
                .len()
                .checked_mul(33)
                .ok_or(Resource::Arithmetic)?,
        )?;
        for signer in policy.effects.signer_identities() {
            signers.push(*signer.as_bytes());
        }
    }
    let mut rows: Vec<NativeConditionalPolicyRootInputV1<'_>> = vector(policies.len(), budget)?;
    budget.charge_work(
        policies
            .len()
            .checked_mul(4 + 32 + 32 + 10 * 32 + 1)
            .ok_or(Resource::Arithmetic)?,
    )?;
    let mut start = 0;
    for (policy, (semantic_root, kernel_binding)) in policies.iter().zip(roots) {
        if semantic_root != policy.semantic_root {
            return Err(Error::Mismatch("original policy/source root order"));
        }
        budget.charge_work(rows.len().checked_mul(36).ok_or(Resource::Arithmetic)?)?;
        if rows
            .iter()
            .any(|row| row.semantic_root == semantic_root || row.kernel_binding == kernel_binding)
        {
            return Err(Error::Mismatch("distinct original conditional roots"));
        }
        let end = start + policy.effects.signer_identities().len();
        rows.push(NativeConditionalPolicyRootInputV1 {
            semantic_root,
            kernel_binding,
            effect_signers: &signers[start..end],
            effect_toolchain: toolchain(policy.effects.toolchain()),
            formula_verifying_key: *policy.formula.verifying_key(),
            formula_toolchain: toolchain(policy.formula.toolchain()),
            formula_boundary: policy.formula.boundary() as u8,
        });
        start = end;
    }
    let input = NativeConditionalPolicyRosterInputV1 {
        source_packet: packet,
        roots: &rows,
    };
    let layout = NativeConditionalPolicyRosterLayoutV1::new(input, |w| budget.charge_work(w))
        .map_err(Error::PolicyRoster)?;
    budget.reserve_storage(
        size_of::<Vec<u8>>()
            .checked_add(layout.encoded_len())
            .ok_or(Resource::Arithmetic)?,
    )?;
    let bytes = encode_native_conditional_policy_roster_v1(input, budget.storage_limit(), |w| {
        budget.charge_work(w)
    })
    .map_err(Error::PolicyRoster)?;
    if bytes.len() != layout.encoded_len() {
        return Err(Resource::Accounting.into());
    }
    budget.reserve_storage(
        bytes
            .capacity()
            .checked_sub(layout.encoded_len())
            .ok_or(Resource::Accounting)?,
    )?;
    Ok(bytes)
}

pub(super) fn toolchain(value: VerusToolchainIdentityV2) -> [[u8; 32]; 5] {
    [
        *value.verus_executable().as_bytes(),
        *value.verus_configuration().as_bytes(),
        *value.solver_executable().as_bytes(),
        *value.solver_configuration().as_bytes(),
        *value.runtime_closure().as_bytes(),
    ]
}

pub(super) fn vector<T>(count: usize, budget: &mut Budget<'_>) -> Result<Vec<T>, Error> {
    let requested = count
        .checked_mul(size_of::<T>())
        .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(
        requested
            .checked_add(size_of::<Vec<T>>())
            .ok_or(Resource::Arithmetic)?,
    )?;
    budget.charge_work(1)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Resource::Allocation)?;
    let actual = values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(actual.checked_sub(requested).ok_or(Resource::Accounting)?)?;
    Ok(values)
}

#[cfg(test)]
#[path = "production_pipeline_conditional_policy_capture_v1_tests.rs"]
mod tests;
