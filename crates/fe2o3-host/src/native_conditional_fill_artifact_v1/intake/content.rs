//! Real source recovery and exact finalizer replay, without publication promotion.
use super::*;
use fe2o3_artifact_transaction::{
    InertCompilerExecutionSubjectV3 as Subject, WorkerV3ExternalProviderPayloadsV1 as Providers,
};
use fe2o3_compiler_execution_protocol::CompilerExecutionIssuerPolicyV3 as Policy;
use fe2o3_compiler_ffi::{
    INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5 as METADATA,
    InertSemanticCompilerModuleHandoffV5 as Handoff,
    inert_semantic_compiler_module_handoff_decode_work_v5 as decode_work,
};
use fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1 as Expected;
use fe2o3_hsaco_finalize::{
    derive_recovered_conditional_worker_publication_intent_in_original_account_v5 as derive_intent,
    revalidate_conditional_worker_finalizer_in_original_account_v5 as replay,
};
use fe2o3_verifier::recover_native_conditional_handoff_under_policy_file_v1 as recover_source;
use fe2o3_verifier::recover_native_conditional_handoff_under_policy_file_with_cpu_mapping_v1 as recover_mapped_source;

#[path = "content_enrollment.rs"]
mod enrollment;

// The enclosing intake owns the prepaid envelope and independently revalidated
// profiles. This move-only intermediate preserves their original account/floor;
// it is not publication custody, currentness, or launch authority.
pub(super) struct AuthenticatedInput<'work> {
    handoff: Handoff,
    carriage: Carriage,
    enrollment: Option<Expected>,
    ledger: Ledger,
    account: Account,
    floor: usize,
    _work: std::marker::PhantomData<&'work fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1>,
}

pub(super) fn authenticate<'work>(
    wire: &Wire<'_>,
    record: &Record,
    compiler: &Compiler<'work>,
    profile: &Profile<'work>,
    budget: &mut Budget<'work>,
) -> Result<AuthenticatedInput<'work>> {
    budget.check_prior_denials_v1()?;
    let account = budget
        .storage_account_identity_v1()
        .ok_or(Resource::Accounting)?;
    let ledger = budget.work_ledger_identity_v1();
    budget.reserve_storage(size_of::<AuthenticatedInput<'work>>())?;
    let (carriage, charge) =
        Carriage::decode_in_original_account_v3(wire.compiler_execution_bytes(), budget)
            .map_err(failure)?;
    budget.reserve_storage(charge.additional_storage())?;
    require_same_policy(
        compiler.policy(),
        carriage.policy(),
        &profile.configuration().parts().compiler_policy_identity,
        budget,
    )?;

    exact(
        wire.outer_handoff_bytes(),
        record.outer_handoff_sha256(),
        record.outer_handoff_length(),
        budget,
    )?;
    budget.charge_work(
        decode_work(wire.outer_handoff_bytes().len())
            .map_err(|error| failure(format_args!("native V5 decode bound: {error:?}")))?,
    )?;
    budget.reserve_storage(METADATA)?;
    let bytes = copy(wire.outer_handoff_bytes(), budget)?;
    let handoff = Handoff::decode_owned(bytes)
        .map_err(|error| failure(format_args!("native V5 handoff: {error:?}")))?;
    authenticate_raw_subject(&handoff, carriage.request().subject(), budget)?;
    let floor = budget.storage();
    require_original_account(ledger, account, floor, budget)?;
    let mut authenticated = AuthenticatedInput {
        handoff,
        carriage,
        enrollment: None,
        ledger,
        account,
        floor,
        _work: std::marker::PhantomData,
    };
    authenticated.require_enrollment(compiler.policy(), budget)?;
    Ok(authenticated)
}

fn require_same_policy(
    independent: &Policy,
    carried: &Policy,
    configured_identity: &[u8; 32],
    budget: &mut Budget<'_>,
) -> Result<()> {
    // Both nominal policies were strictly decoded, including their computed ID.
    // Pay all comparisons even when the first mismatch would short-circuit.
    let work = independent
        .canonical_bytes()
        .len()
        .checked_add(2 * size_of::<[u8; 32]>() + size_of::<u64>())
        .ok_or(Resource::Arithmetic)?;
    budget.charge_work(work)?;
    require(
        independent.canonical_bytes() == carried.canonical_bytes()
            && independent.identity() == carried.identity()
            && independent.generation() == carried.generation()
            && independent.identity().as_bytes() == configured_identity,
        "native independently pinned compiler policy",
    )
}

fn require_original_account(
    ledger: Ledger,
    account: Account,
    floor: usize,
    budget: &mut Budget<'_>,
) -> Result<()> {
    budget.check_prior_denials_v1()?;
    budget.charge_work(4)?;
    if budget.work_ledger_identity_v1() != ledger
        || budget.storage_account_identity_v1() != Some(account)
        || budget.storage() < floor
    {
        return Err(Resource::Accounting.into());
    }
    Ok(())
}

fn authenticate_raw_subject(
    handoff: &Handoff,
    claimed: &Subject,
    budget: &mut Budget<'_>,
) -> Result<()> {
    // These are compiler receipt coordinates, not the Worker's publication attempt.
    let coordinates = size_of::<(
        fe2o3_artifact_transaction::BuildAttempt,
        fe2o3_artifact_transaction::CompilerModuleHandoffSlotV5,
        fe2o3_artifact_transaction::CompilerModuleHandoffTransactionIdentityV5,
    )>();
    budget.reserve_storage(coordinates)?;
    let (subject, charge) = Subject::from_replay_evidence_in_original_account_v3(
        claimed.attempt(),
        claimed.slot(),
        claimed.transaction_identity(),
        handoff,
        budget,
    )
    .map_err(failure)?;
    budget.reserve_storage(charge.retained_storage())?;
    require_same_subject(&subject, claimed, budget)?;
    drop(subject);
    budget.release_storage(charge.retained_storage())?;
    budget.release_storage(coordinates)?;
    Ok(())
}

fn require_same_subject(
    subject: &Subject,
    claimed: &Subject,
    budget: &mut Budget<'_>,
) -> Result<()> {
    budget.charge_work(subject.canonical_bytes().len())?;
    require(
        subject.canonical_bytes() == claimed.canonical_bytes(),
        "native raw V5 compiler-execution subject",
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recover<'work>(
    authenticated: AuthenticatedInput<'work>,
    wire: &Wire<'_>,
    record: &Record,
    claim: &Claim,
    publication: &Publication<'work>,
    producer: &ProducerIdentity,
    profile: &Profile<'work>,
    budget: &mut Budget<'work>,
) -> Result<(Finalized, Transcript, Carriage)> {
    require_original_account(
        authenticated.ledger,
        authenticated.account,
        authenticated.floor,
        budget,
    )?;
    let AuthenticatedInput {
        handoff,
        carriage,
        enrollment,
        ..
    } = authenticated;
    exact(
        wire.transcript_bytes(),
        record.transcript_sha256(),
        record.transcript_length(),
        budget,
    )?;
    exact(
        publication.exact_artifact_bytes(),
        record.output_sha256(),
        record.output_length(),
        budget,
    )?;

    let (source, charge) = match enrollment {
        None => recover_source(profile.semantic_policy_bytes(), handoff, budget),
        Some(expected) => {
            recover_mapped_source(profile.semantic_policy_bytes(), handoff, &expected, budget)
        }
    }
    .map_err(failure)?;
    budget.reserve_storage(charge.retained_storage())?;

    let (transcript, charge) =
        Transcript::decode_canonical(wire.transcript_bytes(), budget).map_err(failure)?;
    budget.reserve_storage(charge.retained_storage())?;

    let count = wire.providers().len();
    budget.reserve_storage(
        count
            .checked_mul(size_of::<Vec<u8>>())
            .and_then(|n| n.checked_add(size_of::<Providers>() + size_of::<Vec<Vec<u8>>>()))
            .ok_or(Resource::Arithmetic)?,
    )?;
    let mut providers = Vec::new();
    providers.try_reserve_exact(count).map_err(failure)?;
    require(providers.capacity() == count, "native provider capacity")?;
    for bytes in wire.providers() {
        providers.push(copy(bytes, budget)?);
    }
    let providers = Providers::new(providers).map_err(failure)?;
    require(
        providers.len() == record.external_provider_count()
            && providers.payload_length() == record.external_provider_payload_length()
            && providers.canonical_length() == record.external_provider_archive_length()
            && providers.canonical_sha256() == record.external_provider_archive_sha256(),
        "native exact provider archive",
    )?;
    let (finalized, charge) = replay(
        producer,
        record.attempt(),
        source,
        &transcript,
        providers.into_payloads(),
        publication.exact_artifact_bytes(),
        budget,
    )
    .map_err(failure)?;
    budget.reserve_storage(charge.retained_storage())?;

    // This adapter only derives the inert plan from actual recovered custody.
    // It cannot rebrand the source as fresh consumption or publish a transaction.
    let (intent, charge) =
        derive_intent(producer, &finalized, &transcript, budget).map_err(failure)?;
    budget.reserve_storage(charge.retained_storage())?;
    require(
        intent.durable_plan() == record.plan() && intent.durable_plan() == claim.plan(),
        "native exact replay-derived publication plan",
    )?;
    budget.release_storage(charge.retained_storage())?;
    let source = finalized.source();
    let (subject, charge) = Subject::from_publication_in_original_account_v3(
        source.binding().receipt(),
        source.recovered_handoff().handoff(),
        budget,
    )
    .map_err(failure)?;
    budget.reserve_storage(charge.retained_storage())?;
    budget.charge_work(subject.canonical_bytes().len())?;
    require(
        subject.canonical_bytes() == carriage.request().subject().canonical_bytes(),
        "native actual V5 compiler-execution subject",
    )?;
    drop(subject);
    budget.release_storage(charge.retained_storage())?;
    Ok((finalized, transcript, carriage))
}

fn exact(bytes: &[u8], sha: [u8; 32], length: usize, budget: &mut Budget<'_>) -> Result<()> {
    budget.charge_work(bytes.len().checked_add(64).ok_or(Resource::Arithmetic)?)?;
    require(
        bytes.len() == length && <[u8; 32]>::from(Sha256::digest(bytes)) == sha,
        "native exact readiness component",
    )
}

fn copy(bytes: &[u8], budget: &mut Budget<'_>) -> Result<Vec<u8>> {
    budget.charge_work(bytes.len().checked_mul(4).ok_or(Resource::Arithmetic)?)?;
    budget.reserve_storage(
        bytes
            .len()
            .checked_add(size_of::<Vec<u8>>())
            .ok_or(Resource::Arithmetic)?,
    )?;
    let mut owned = Vec::new();
    owned.try_reserve_exact(bytes.len()).map_err(failure)?;
    require(owned.capacity() == bytes.len(), "native component capacity")?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

    #[test]
    fn exact_component_rejects_independent_length_and_digest_substitution() {
        let bytes = b"exact original component";
        let hash: [u8; 32] = Sha256::digest(bytes).into();
        for (sha, length, ok) in [
            (hash, bytes.len(), true),
            ([0; 32], bytes.len(), false),
            (hash, bytes.len() - 1, false),
        ] {
            let mut work = Work::new(usize::MAX);
            let mut budget = Budget::new(&mut work, 4096);
            assert_eq!(exact(bytes, sha, length, &mut budget).is_ok(), ok);
            assert_eq!(budget.storage(), 0);
            assert_eq!(budget.work(), bytes.len() + 64);
        }
        let mut work = Work::new(0);
        let mut budget = Budget::new(&mut work, 4096);
        assert!(matches!(
            exact(bytes, hash, bytes.len(), &mut budget),
            Err(Error::Resource(_))
        ));
    }
}

#[cfg(test)]
#[path = "content_gate_tests.rs"]
mod gate_tests;
