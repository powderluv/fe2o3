//! Consuming native execution on the original root account and deadline.
use super::*;
use crate::compiler_invocation_backing::{
    CompilerInvocationBacking as Backing, CompilerInvocationBackingError as BackingError,
};
use crate::proof_helper_backing::{ProofHelperBacking, ProofHelperBackingError};
use crate::proof_helper_launch::{self, ManagedProofHelper, ProofHelperLaunchError};
use fe2o3_compiler_closure_capability::{
    ApprovedCompilerPolicyV2 as Approval, CompilerApprovalErrorV2 as ApprovalError,
    RetainedCompilerRuntimeErrorV1 as RuntimeError, RetainedCompilerRuntimeV1 as Runtime,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
    CanonicalKernelIrWorkLedgerIdentityV1 as Ledger,
};
use fe2o3_protected_service_profile::ProtectedServiceCredentialProfileV1 as Credentials;
use fe2o3_protected_service_spawn::ProtectedServiceCleanupServiceV2 as Cleanup;
use std::sync::Arc;

#[path = "native_compiler_attempt.rs"]
mod compiler_attempt;
pub(crate) use compiler_attempt::CompilerConfinement;

#[path = "native_root_request_quota.rs"]
mod quota;

#[path = "native_root_request_runtime.rs"]
mod execution;

#[path = "native_root_request_enrollment.rs"]
mod enrollment;

#[cfg(test)]
#[path = "native_root_request_fault_tests.rs"]
mod faults;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Received,
    Prepared,
    HelperReady,
    CompilerGated,
    Interrupting,
    Interrupted,
    Armed,
    AwaitingExec,
    FirstExec,
    HeldExec,
    ConfirmedExec,
    Running,
    RootExitHeld,
    PublicationObserved,
    AwaitingTerminal,
    Retiring,
    CompletionReady,
    DrainingCompletion,
    Completed,
    Failed,
    Cancelled,
}

/// Owns the ORIGINAL preparation and complete receiver before any compiler
/// preparation can fail. No constructor accepts a descriptor, runtime approval,
/// substitute Prepared, or a reconstructed transport identity.
pub(crate) struct RootCompilerRequest<'work> {
    // Cancel the anchor before receiver/backing retirement on unwind too.
    prepared: Option<Prepared>,
    receiver: Arc<Receiver>,
    backing: Option<Backing>,
    helper: Option<ManagedProofHelper>,
    attempt: Option<compiler_attempt::Attempt<'work>>,
    terminal: Option<execution::Termination>,
    completion: Option<execution::Completion>,
    // Outer None is unjoined; inner None is the original no-enrollment case.
    enrollment: Option<Option<fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1>>,
    state: State,
    ledger: Ledger,
    address: usize,
    reserved: usize,
}

impl<'work> RootCompilerRequest<'work> {
    pub(crate) const fn completion_was_published(&self) -> bool {
        matches!(self.state, State::Completed)
    }

    pub(crate) fn original_completion_termination(&self) -> Result<execution::Termination> {
        self.completion
            .as_ref()
            .map(|record| record.terminal().termination())
            .ok_or_else(|| rejected("missing original terminal completion"))
    }

    // Prepared and Receiver already have full reservations. The new optional
    // backing is inline here; its heap/file growth is charged by its constructor.
    pub(crate) const ENVELOPE: usize =
        size_of::<Self>() - size_of::<Prepared>() + 2 * size_of::<usize>();
    const LOCAL_WORK: usize = root::LOCAL_WORK + Backing::LOCAL_WORK;

    pub(crate) fn preparation_quota() -> Result<root::CompilerExecutionRootAdmissionQuotaV2> {
        quota::preparation()
    }

    pub(crate) fn refusal_quota() -> Result<root::CompilerExecutionRootAdmissionQuotaV2> {
        quota::refusal()
    }

    pub(crate) fn launch_quota() -> Result<root::CompilerExecutionRootAdmissionQuotaV2> {
        quota::launch()
    }

    pub(crate) fn cleanup_growth() -> Result<(usize, usize)> {
        quota::cleanup_growth()
    }

    pub(crate) fn cancellation_quota() -> Result<root::CompilerExecutionRootAdmissionQuotaV2> {
        let trace = compiler_attempt::Attempt::cancellation_quota().map_err(helper_error)?;
        Ok(root::CompilerExecutionRootAdmissionQuotaV2 {
            work: root::sum(&[root::LOCAL_WORK, trace.work()])?,
            scratch: trace.scratch(),
        })
    }

    /// Infallible ownership installation, BEFORE any fallible validation or
    /// admission. Native prepaid ENVELOPE alongside the whole Receiver before
    /// accept. Preserve every existing reservation until this request drops.
    pub(crate) fn install(prepared: Prepared, receiver: Receiver, b: &Budget<'_>) -> Self {
        Self {
            prepared: Some(prepared),
            receiver: Arc::new(receiver),
            backing: None,
            helper: None,
            attempt: None,
            terminal: None,
            completion: None,
            enrollment: None,
            state: State::Received,
            ledger: b.work_ledger_identity_v1(),
            address: b as *const Budget<'_> as usize,
            reserved: b.storage(),
        }
    }

    pub(crate) fn continuity(&self, b: &mut Budget<'_>) -> Result<()> {
        self.check_account(b)?;
        if let Some(prepared) = &self.prepared {
            prepared.revalidate(b)?;
        } else {
            self.check_complete(b)?;
            // Once the checked terminal syscall is released, live-process
            // validation is no longer possible. The original controller still
            // owes the actual root wait and every acquired task's retirement.
            if !matches!(
                self.state,
                State::AwaitingTerminal | State::Retiring | State::CompletionReady
            ) {
                self.attempt()?.continuity(b).map_err(helper_error)?;
            }
        }
        Ok(())
    }

    /// Native supplies its original creator's pool. Helper exec is permitted by
    /// that dedicated deployment contract. Only the original native attempt can
    /// release its gate, validate first exec and control subsequent checkpoints.
    /// True requires retaining the original terminal publication completion;
    /// delivery is separately gated on the original aggregate cleanup. All
    /// intermediate states return false and still require another turn.
    ///
    /// # Safety
    /// Preserve the entrypoint's actual outside whole-domain custodian, unique
    /// creator/reaper and mutation exclusions through aggregate pool shutdown.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn step(
        &mut self,
        cleanup: &mut Cleanup,
        b: &mut Budget<'work>,
    ) -> Result<bool> {
        let state = self.state;
        // Set before even entry charging: failure AND unwind consume this turn.
        self.state = State::Failed;
        b.charge_work(Self::LOCAL_WORK)?;
        self.check_account(b)?;
        self.check_complete(b)?;
        match state {
            State::Received => {
                self.prepare(b)?;
                self.check_complete(b)?;
                self.state = State::Prepared;
                Ok(false)
            }
            State::Prepared => {
                self.require_cleanup_guard(cleanup, b)?;
                let backing = self
                    .backing
                    .take()
                    .ok_or_else(|| rejected("missing original backing"))?;
                let (backing, growth) =
                    ProofHelperBacking::prepare(backing, b).map_err(helper_backing_error)?;
                self.reserve_growth(growth, b)?;
                let peer =
                    net::sockopt::socket_peercred(self.receiver.connection.as_ref().unwrap())
                        .map_err(|source| Error::Io {
                            operation: "original compiler peer",
                            source,
                        })?;
                if self.receiver.sender
                    != Some(io::MessageSender::new(
                        peer.pid.as_raw_pid(),
                        peer.uid.as_raw(),
                        peer.gid.as_raw(),
                    ))
                {
                    return Err(rejected("original compiler peer changed"));
                }
                let credentials =
                    fe2o3_protected_service_profile::ProtectedServiceCredentialProfileV1::new(
                        peer.uid.as_raw(),
                        peer.gid.as_raw(),
                    )
                    .map_err(|_| rejected("compiler peer credentials refused"))?;
                require_separate_helper_peer(backing.credentials(), credentials)?;
                let deadline = self.receiver.deadline.unwrap();
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| rejected("intake deadline exceeded"))?;
                let session = *self.receiver.challenge.as_ref().unwrap().identity();
                // The additional shared handle is a new payload field, not a
                // second reservation of the receiver or consumed capture.
                self.reserve_growth(size_of::<Arc<Receiver>>(), b)?;
                // SAFETY: the original complete receiver is published in the
                // helper's funded payload before clone. Native supplies the sole
                // dedicated creator and actual outside-domain custody contract.
                let (helper, growth) = unsafe {
                    proof_helper_launch::launch(
                        backing,
                        Arc::clone(&self.receiver),
                        session,
                        credentials,
                        remaining,
                        cleanup,
                        b,
                    )
                }
                .map_err(helper_error)?;
                self.helper = Some(helper);
                self.reserve_growth(growth, b)?;
                #[cfg(test)]
                Self::postclone_checkpoint_for_test(
                    "helper-ready",
                    self.helper.as_ref().unwrap().pid_for_test(),
                    b,
                )?;
                self.state = State::HelperReady;
                Ok(false)
            }
            State::HelperReady => {
                self.require_cleanup_guard(cleanup, b)?;
                let peer =
                    net::sockopt::socket_peercred(self.receiver.connection.as_ref().unwrap())
                        .map_err(|source| Error::Io {
                            operation: "original compiler peer",
                            source,
                        })?;
                if self.receiver.sender
                    != Some(io::MessageSender::new(
                        peer.pid.as_raw_pid(),
                        peer.uid.as_raw(),
                        peer.gid.as_raw(),
                    ))
                {
                    return Err(rejected("original compiler peer changed"));
                }
                let credentials =
                    fe2o3_protected_service_profile::ProtectedServiceCredentialProfileV1::new(
                        peer.uid.as_raw(),
                        peer.gid.as_raw(),
                    )
                    .map_err(|_| rejected("compiler peer credentials refused"))?;
                let helper = self
                    .helper
                    .take()
                    .ok_or_else(|| rejected("missing original helper"))?;
                // SAFETY: same dedicated creator/pool as helper launch; only the
                // original receiver supplies cwd/stdio. The gate stays closed.
                let (attempt, growth) = unsafe {
                    compiler_attempt::launch(
                        helper,
                        &self.receiver,
                        credentials,
                        self.receiver.deadline.unwrap(),
                        cleanup,
                        b,
                    )
                }
                .map_err(helper_error)?;
                self.attempt = Some(attempt);
                self.reserve_growth(growth, b)?;
                self.state = State::CompilerGated;
                Ok(false)
            }
            // SAFETY: exactly the same original creator, pool, account and
            // outside-domain custody supplied to this one consuming request.
            _ => unsafe { self.execution_step(state, cleanup, b) },
        }
    }

    fn require_cleanup_guard(&self, cleanup: &mut Cleanup, b: &mut Budget<'_>) -> Result<()> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| rejected("missing original preparation"))?;
        // This existing join validates the actual pool guard against the retained
        // root-bound lifecycle; the same request and cleanup accounts fund it.
        prepared.validate_cleanup_guard(cleanup, b)?;
        Ok(())
    }

    fn prepare(&mut self, b: &mut Budget<'_>) -> Result<()> {
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| rejected("missing original preparation"))?;
        let floor = root::sum(&[
            prepared.retained_storage(),
            Receiver::STORAGE,
            Self::ENVELOPE,
        ])?;
        if b.storage() < floor {
            return Err(Resource::Accounting.into());
        }
        let (approval, charge) = Approval::from_production_policy(b).map_err(approval_error)?;
        self.reserve_growth(charge.retained_storage(), b)?;
        let prepared = self.prepared.as_ref().unwrap();
        let profile = approval.profile().profile();
        let deployment = prepared.trust.deployment().deployment();
        if profile.policy().canonical_bytes() != prepared.trust.policy().policy().canonical_bytes()
            || profile.supervisor_uid() != deployment.service_uid()
            || profile.supervisor_gid() != deployment.service_gid()
            || profile.external_anchor_service() != deployment.external_anchor_service()
        {
            return Err(rejected(
                "fixed compiler approval differs from original root",
            ));
        }
        let (runtime, charge) =
            Runtime::from_production_runtime(approval, b).map_err(runtime_error)?;
        self.reserve_growth(charge.additional_storage(), b)?;
        // Both are the original decoder/capture owners, not reconstructions from
        // the retained raw rights. Failed consumption never attempts to recover them.
        let receiver = Arc::get_mut(&mut self.receiver)
            .ok_or_else(|| rejected("received inputs were already transferred"))?;
        let capture = receiver
            .invocation
            .take()
            .ok_or_else(|| rejected("missing received invocation"))?;
        let output = receiver
            .output
            .take()
            .ok_or_else(|| rejected("missing received output"))?;
        let (backing, charge) =
            Backing::prepare(runtime, capture, output, b).map_err(backing_error)?;
        self.reserve_growth(charge.additional_storage(), b)?;
        self.backing = Some(backing);
        Ok(())
    }

    fn reserve_growth(&mut self, growth: usize, b: &mut Budget<'_>) -> Result<()> {
        let reserved = root::sum(&[self.reserved, growth])?;
        b.reserve_storage(growth)?;
        self.reserved = reserved;
        Ok(())
    }

    fn check_account(&self, b: &Budget<'_>) -> Result<()> {
        if b.work_ledger_identity_v1() != self.ledger
            || b as *const Budget<'_> as usize != self.address
            || b.storage() < self.reserved
        {
            return Err(Resource::Accounting.into());
        }
        Ok(())
    }

    fn check_complete(&self, b: &mut Budget<'_>) -> Result<()> {
        if self.receiver.phase != Phase::Ack
            || self.receiver.connection.is_none()
            || self.receiver.sender.is_none()
            || self.receiver.ack.is_none()
            || self.receiver.last.is_none()
        {
            return Err(rejected(
                "compiler request requires complete original intake",
            ));
        }
        let challenge = self
            .receiver
            .challenge
            .as_ref()
            .ok_or_else(|| rejected("missing original challenge"))?;
        let policy = match &self.prepared {
            Some(prepared) => *prepared.trust.policy().policy().identity().as_bytes(),
            None => self
                .attempt()?
                .original_policy_identity(b)
                .map_err(helper_error)?,
        };
        if challenge.policy_identity() != &policy {
            return Err(rejected(
                "received request differs from original root policy",
            ));
        }
        let count = challenge.roles().count();
        if self.receiver.files[..count].iter().any(Option::is_none)
            || self.receiver.files[count..].iter().any(Option::is_some)
        {
            return Err(rejected("complete intake lost an original received right"));
        }
        if self
            .receiver
            .deadline
            .is_none_or(|deadline| Instant::now() >= deadline)
        {
            return Err(rejected("intake deadline exceeded"));
        }
        Ok(())
    }

    pub(crate) fn cancel(&mut self) {
        self.state = if matches!(
            self.state,
            State::CompletionReady | State::DrainingCompletion
        ) {
            State::DrainingCompletion
        } else {
            State::Cancelled
        };
        if let Some(attempt) = &mut self.attempt {
            let _ = attempt.cancel();
            if !attempt.needs_foreground_cancellation() {
                drop(self.attempt.take());
            }
        }
        if let Some(helper) = self.helper.take() {
            let _ = helper.cancel();
        }
        // The original pool must reap the anchor before signal restoration.
        // Raw received rights and any prepared compiler backing stay through drain.
        drop(self.prepared.take());
    }

    /// Retire only foreground wait ownership. A true result still requires the
    /// original independently funded pool to finish every deferred domain.
    pub(crate) fn cancel_step(&mut self, b: &mut Budget<'_>) -> Result<bool> {
        if !matches!(self.state, State::Cancelled | State::DrainingCompletion) {
            return Err(rejected("foreground cleanup requires cancellation"));
        }
        if self.attempt.is_none() {
            return Ok(true);
        }
        b.charge_work(root::LOCAL_WORK)?;
        self.check_account(b)?;
        if let Some(attempt) = &mut self.attempt {
            if !attempt.cancel_step(b).map_err(helper_error)? {
                return Ok(false);
            }
            drop(self.attempt.take());
        }
        Ok(true)
    }

    #[cfg(test)]
    pub(crate) fn arm_postclone_fault_for_test(
        &self,
        phase: &'static str,
        action: &'static str,
        b: &Budget<'_>,
    ) {
        assert!(matches!(self.state, State::Prepared | State::HelperReady));
        faults::arm(&self.receiver, phase, action, b);
    }

    #[cfg(test)]
    pub(crate) fn postclone_fault_observation_for_test() -> faults::Observation {
        faults::observation()
    }

    #[cfg(test)]
    pub(crate) fn postclone_checkpoint_for_test(
        phase: &'static str,
        pid: rustix::process::Pid,
        b: &mut Budget<'_>,
    ) -> std::result::Result<(), Resource> {
        faults::checkpoint(phase, pid, b)
    }

    #[cfg(test)]
    pub(crate) fn received_for_test(&self) -> (&Receiver, Option<&Backing>) {
        (&self.receiver, self.backing.as_ref())
    }

    #[cfg(test)]
    pub(crate) fn failed_for_test(&self) -> bool {
        self.state == State::Failed
    }

    #[cfg(test)]
    pub(crate) fn helper_pid_for_test(&self) -> Option<rustix::process::Pid> {
        (self.state == State::HelperReady).then(|| self.helper.as_ref().unwrap().pid_for_test())
    }

    #[cfg(test)]
    pub(crate) fn compiler_pid_for_test(&self) -> Option<rustix::process::Pid> {
        (self.state == State::CompilerGated).then(|| self.attempt.as_ref().unwrap().pid())
    }
}

fn require_separate_helper_peer(helper: Credentials, peer: Credentials) -> Result<()> {
    if helper.uid() == peer.uid() || helper.gid() == peer.gid() {
        return Err(rejected(
            "proof helper and original compiler peer credentials overlap",
        ));
    }
    Ok(())
}

fn helper_backing_error(error: ProofHelperBackingError) -> Error {
    match error {
        ProofHelperBackingError::Resource(e) => e.into(),
        ProofHelperBackingError::Compiler(e) => backing_error(e),
        ProofHelperBackingError::Runtime(e) => runtime_error(e),
        ProofHelperBackingError::Image(
            fe2o3_protected_static_executable::ProtectedStaticExecutableErrorV2::Resource(e),
        ) => e.into(),
        _ => rejected("proof helper backing refused"),
    }
}

fn helper_error(error: ProofHelperLaunchError) -> Error {
    match error {
        ProofHelperLaunchError::Resource(e) => e.into(),
        ProofHelperLaunchError::Backing(e) => helper_backing_error(e),
        ProofHelperLaunchError::Runtime(e) => runtime_guard_error(e),
        ProofHelperLaunchError::Descriptor(crate::native_runtime_descriptors::Error::Resource(
            e,
        )) => e.into(),
        ProofHelperLaunchError::Inventory(crate::native_runtime_inventory::Error::Resource(e)) => {
            e.into()
        }
        ProofHelperLaunchError::Inventory(crate::native_runtime_inventory::Error::Backing(e)) => {
            backing_error(e)
        }
        ProofHelperLaunchError::Native(
            crate::native_launch::CompilerExecutionLaunchErrorV2::Resource(e),
        ) => e.into(),
        ProofHelperLaunchError::Native(e) => e.into(),
        ProofHelperLaunchError::Record(
            fe2o3_compiler_execution_protocol::ProofExecutorBootstrapErrorV1::Resource(e),
        ) => e.into(),
        _ => rejected("native pre-exec attempt refused"),
    }
}

fn runtime_guard_error(error: crate::native_runtime_guard::Error) -> Error {
    use crate::native_runtime_guard::Error as Guard;
    use fe2o3_protected_service_spawn::native_spawn::ProtectedServiceSpawnErrorV2 as Spawn;
    match error {
        Guard::Resource(e)
        | Guard::Spawn(Spawn::Resource(e))
        | Guard::Inventory(crate::native_runtime_inventory::Error::Resource(e))
        | Guard::Descriptor(crate::native_runtime_descriptors::Error::Resource(e)) => e.into(),
        Guard::Inventory(crate::native_runtime_inventory::Error::Backing(e)) => backing_error(e),
        _ => rejected("native runtime checkpoint refused"),
    }
}

fn approval_error(error: ApprovalError) -> Error {
    match error {
        ApprovalError::Resource(e) => e.into(),
        ApprovalError::Capability(e) => e.into(),
        ApprovalError::Codec(fe2o3_build_authority::CompilerApprovalPolicyErrorV2::Framing(
            fe2o3_build_authority::CompilerApprovalPolicyErrorV1::Charge(e),
        )) => e.into(),
        ApprovalError::Mismatch(reason) => root::invalid("compiler approval", reason),
        _ => rejected("fixed-origin compiler approval refused"),
    }
}

fn runtime_error(error: RuntimeError) -> Error {
    match error {
        RuntimeError::Resource(e) => e.into(),
        RuntimeError::Approval(e) => approval_error(e),
        RuntimeError::Capability(e) => e.into(),
        RuntimeError::Codec(fe2o3_build_authority::CompilerRuntimeManifestErrorV1::Charge(e)) => {
            e.into()
        }
        RuntimeError::Mismatch(reason) => root::invalid("compiler runtime", reason),
        _ => rejected("fixed-origin compiler runtime refused"),
    }
}

fn backing_error(error: BackingError) -> Error {
    use crate::compiler_invocation_staging::RustcInvocationStagingErrorV1 as StagingError;
    match error {
        BackingError::Resource(e) | BackingError::Staging(StagingError::Resource(e)) => e.into(),
        BackingError::Capture(e) => e.into(),
        BackingError::Runtime(e) => runtime_error(e),
        BackingError::Output(e) => output_error(e),
        _ => rejected("received compiler invocation preparation refused"),
    }
}

#[cfg(test)]
#[path = "native_root_request_tests.rs"]
mod tests;
