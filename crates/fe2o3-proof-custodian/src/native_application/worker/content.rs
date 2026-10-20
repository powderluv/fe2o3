//! Nested original V5/analyzer/refinement ownership; wire evidence is never promoted.
use super::*;
use fe2o3_artifact_transaction::InertCompilerExecutionSubjectV3 as Subject;
use fe2o3_compiler_ffi::{
    INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5,
    InertSemanticCompilerModuleHandoffV5 as Handoff,
    inert_semantic_compiler_module_handoff_decode_work_v5,
};
use fe2o3_kernel_analysis::{
    PhysicalMachineEffectBudgetV1 as MachineBudget, PhysicalMachineEffectEntryRequestV1 as Entry,
};
use fe2o3_runtime_protocol::{
    InertConditionalWorkerReadinessWireV5 as Wire,
    NativeConditionalApplicationBindingV1 as Association,
};
use fe2o3_verifier::{
    check_native_conditional_fill_program_v1, execute_native_conditional_fill_refinement_v1,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<'work>(
    application: &ActiveNativeApplication<'work>,
    session: &Session,
    root: &wire::ControlEndpoint,
    parent: (i32, u32, u32),
    policy: &Policy<'work>,
    tools: &Tools,
    config: &Config,
    inputs: &Inputs,
    readiness_file: &File,
    payload_file: &File,
    deadline: Instant,
    budget: &mut Budget<'work>,
) -> io::Result<()> {
    let sender = super::sender(application);
    let (readiness, s) = read_sealed_file(
        readiness_file,
        (sender.1, sender.2),
        MAX_CONDITIONAL_WORKER_READINESS_BYTES_V5,
        budget,
    )?;
    budget.reserve_storage(s).map_err(other)?;
    let (payload, s) = read_sealed_file(
        payload_file,
        (sender.1, sender.2),
        NATIVE_APPLICATION_PROOF_MAX_PAYLOAD_BYTES_V1,
        budget,
    )?;
    budget.reserve_storage(s).map_err(other)?;
    budget
        .charge_work(readiness.len() + payload.len())
        .map_err(other)?;
    require(
        digest(&readiness) == inputs.readiness() && digest(&payload) == inputs.payload(),
        "native proof sealed inputs differ",
    )?;
    budget.charge_work(4 * readiness.len()).map_err(other)?;
    budget
        .reserve_storage(size_of::<Wire<'_>>() + 4096)
        .map_err(other)?;
    let wire =
        Wire::decode(&readiness, MAX_CONDITIONAL_WORKER_READINESS_BYTES_V5).map_err(other)?;
    let (carriage, s) =
        Carriage::decode_in_original_account_v3(wire.compiler_execution_bytes(), budget)
            .map_err(other)?;
    budget
        .reserve_storage(s.additional_storage())
        .map_err(other)?;
    let registration = &application.capsule().binding;
    let association = Association::bind(
        &readiness,
        registration.compiler_handoff(),
        &carriage,
        budget,
    )
    .map_err(other)?;
    budget
        .reserve_storage(size_of::<Association>())
        .map_err(other)?;
    require(
        association.canonical_bytes() == registration.association().canonical_bytes()
            && *carriage.policy().identity().as_bytes() == config.compiler_policy_identity(),
        "native content registration/policy association differs",
    )?;
    let outer = wire.outer_handoff_bytes();
    let begin = outer.as_ptr() as usize - readiness.as_ptr() as usize;
    let range = begin..begin + outer.len();
    budget
        .charge_work(
            inert_semantic_compiler_module_handoff_decode_work_v5(outer.len())
                .map_err(|_| io::Error::other("native V5 handoff decode bound refused"))?,
        )
        .map_err(other)?;
    budget
        .reserve_storage(
            INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5
                + size_of::<Arc<Vec<u8>>>()
                + size_of::<Vec<u8>>()
                + 2 * size_of::<usize>(),
        )
        .map_err(other)?;
    // Conversion reuses the exact immutable Box backing; Arc holds that allocation.
    let readiness = Arc::new(readiness.into_vec());
    let handoff = Handoff::decode_shared_vec(readiness.clone(), range)
        .map_err(|_| io::Error::other("native V5 outer handoff refused"))?;
    let claimed = carriage.request().subject();
    // Authenticate all raw content before interpreting its semantic evidence.
    let (raw_subject, raw_charge) = join_raw_subject(&handoff, claimed, budget)?;
    drop(raw_subject);
    budget.release_storage(raw_charge).map_err(other)?;
    let (owner, s) = policy.recover(handoff, budget)?;
    budget
        .reserve_storage(s.retained_storage())
        .map_err(other)?;
    let (subject, _) = join_raw_subject(owner.handoff(), claimed, budget)?;
    let (program, s) = check_native_conditional_fill_program_v1(&owner, budget)
        .map_err(|_| io::Error::other("native closed-fill program refused"))?;
    budget
        .reserve_storage(s.retained_storage())
        .map_err(other)?;
    require(Instant::now() < deadline, "native analysis deadline")?;
    let entry =
        Entry::new(program.function_symbol(), MachineBudget::new(2, 1, 1, 1, 0)).map_err(other)?;
    // Existing authenticated analyzer owns the exact input and pinned runtime
    // observation. Its separately bounded transient capture is not a new Budget.
    let analysis = tools
        .analyzer
        .analyze(payload.into_vec(), vec![entry], Tools::analysis_limits()?)
        .map_err(other)?;
    // Charge retained canonical representations before theorem composition; the
    // analyzer's opaque capture implementation remains its explicit trusted domain.
    let analysis_charge = analysis.request().canonical_bytes().len()
        + analysis.analysis().canonical_bytes().len()
        + analysis.canonical_receipt_bytes().len()
        + analysis.request().exact_payload_bytes().len()
        + size_of_val(&analysis);
    budget.reserve_storage(analysis_charge).map_err(other)?;
    tools.revalidate(budget)?;
    policy.revalidate(budget)?;
    require(Instant::now() < deadline, "native proof deadline")?;
    let remaining = deadline
        .saturating_duration_since(Instant::now())
        .as_secs()
        .min(180) as u32;
    require(remaining > 0, "native proof deadline exhausted")?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_native_conditional_fill_refinement_v1(
            &tools.runtime,
            &owner,
            &analysis,
            remaining,
            budget,
        )
    }));
    let (proof, charge) = match result {
        Ok(Ok(value)) => value,
        _ => super::quarantine((
            &owner,
            &analysis,
            &readiness,
            &subject,
            &carriage,
            inputs,
            readiness_file,
            payload_file,
            application,
            policy,
            tools,
            root,
            session,
            &*budget,
        )),
    };
    // Nothing after a possible proof publication may unwind the original borrowed
    // source, analyzer execution or refinement owner. There is no Release message.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<()> {
        budget
            .reserve_storage(charge.retained_storage())
            .map_err(other)?;
        let (evidence, charge) = evidence::describe(inputs, &carriage, &proof, budget)?;
        budget
            .reserve_storage(charge.retained_storage())
            .map_err(other)?;
        let (proved, s) = Message::proved(session, &evidence, budget).map_err(other)?;
        budget
            .reserve_storage(s.retained_storage())
            .map_err(other)?;
        require(
            transport::try_send(application.peer(budget)?, &proved, budget)?,
            "native Proved pending",
        )?;
        drop(proved);
        budget
            .release_storage(s.retained_storage())
            .map_err(other)?;
        super::serve(
            application,
            session,
            root,
            parent,
            tools,
            &evidence,
            |budget| {
                policy.revalidate(budget)?;
                let (observed, s) = evidence::describe(inputs, &carriage, &proof, budget)?;
                budget
                    .reserve_storage(s.retained_storage())
                    .map_err(other)?;
                require(
                    observed.canonical_bytes() == evidence.canonical_bytes(),
                    "native retained proof changed",
                )?;
                drop(observed);
                budget.release_storage(s.retained_storage()).map_err(other)
            },
            budget,
        )
    }));
    let _ = std::hint::black_box(result);
    super::quarantine((
        &proof,
        &owner,
        &analysis,
        &readiness,
        &subject,
        &carriage,
        inputs,
        readiness_file,
        payload_file,
        application,
        policy,
        tools,
        root,
        session,
        &*budget,
    ))
}

/// The caller has already authenticated the claimed subject under the pinned
/// policy. This joins content only; the returned owner is fully charged.
fn join_raw_subject(
    handoff: &Handoff,
    claimed: &Subject,
    budget: &mut Budget<'_>,
) -> io::Result<(Subject, usize)> {
    let coordinates = size_of::<(
        fe2o3_artifact_transaction::BuildAttempt,
        fe2o3_artifact_transaction::CompilerModuleHandoffSlotV5,
        fe2o3_artifact_transaction::CompilerModuleHandoffTransactionIdentityV5,
    )>();
    budget.reserve_storage(coordinates).map_err(other)?;
    let (subject, charge) = Subject::from_replay_evidence_in_original_account_v3(
        claimed.attempt(),
        claimed.slot(),
        claimed.transaction_identity(),
        handoff,
        budget,
    )
    .map_err(other)?;
    let retained = charge.retained_storage();
    budget.reserve_storage(retained).map_err(other)?;
    require_same_subject(&subject, claimed, budget)?;
    budget.release_storage(coordinates).map_err(other)?;
    Ok((subject, retained))
}

fn require_same_subject(
    subject: &Subject,
    claimed: &Subject,
    budget: &mut Budget<'_>,
) -> io::Result<()> {
    budget
        .charge_work(subject.canonical_bytes().len())
        .map_err(other)?;
    require(
        subject.canonical_bytes() == claimed.canonical_bytes(),
        "native V5 subject differs",
    )
}

fn digest(bytes: &[u8]) -> ([u8; 32], u64) {
    (Sha256::digest(bytes).into(), bytes.len() as u64)
}

#[cfg(test)]
#[path = "content_tests.rs"]
mod tests;
