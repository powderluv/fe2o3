//! Original-owner execution transitions. No caller status can complete a request.
use super::*;
use crate::native_runtime_controller::Progress;
use fe2o3_artifact_transaction::MAX_COMPILER_MODULE_HANDOFF_BYTES_V5;
use fe2o3_compiler_execution_protocol::CompilerExecutionRootPublicationCompletionErrorV1 as CompletionError;
pub(super) use fe2o3_compiler_execution_protocol::{
    CompilerExecutionRootPublicationCompletionV1 as Completion,
    CompilerExecutionRootTerminationV1 as Termination,
};

impl<'work> RootCompilerRequest<'work> {
    pub(super) fn attempt(&self) -> Result<&compiler_attempt::Attempt<'work>> {
        self.attempt
            .as_ref()
            .ok_or_else(|| rejected("missing original compiler attempt"))
    }

    fn attempt_mut(&mut self) -> Result<&mut compiler_attempt::Attempt<'work>> {
        self.attempt
            .as_mut()
            .ok_or_else(|| rejected("missing original compiler attempt"))
    }

    /// Preserve step's dedicated creator, original account, cleanup pool and
    /// outside custodian until aggregate terminal cleanup, including unwind.
    #[allow(unsafe_code)]
    pub(super) unsafe fn execution_step(
        &mut self,
        state: State,
        cleanup: &mut Cleanup,
        b: &mut Budget<'work>,
    ) -> Result<bool> {
        self.state = match state {
            State::CompilerGated => {
                self.attempt
                    .as_mut()
                    .ok_or_else(|| rejected("missing original compiler attempt"))?
                    .interrupt_runtime(&self.receiver, b)
                    .map_err(helper_error)?;
                State::Interrupting
            }
            State::Interrupting => {
                if self
                    .attempt_mut()?
                    .poll_runtime_interrupt(b)
                    .map_err(helper_error)?
                {
                    State::Interrupted
                } else {
                    State::Interrupting
                }
            }
            State::Interrupted => {
                // SAFETY: the same original closed gate and custodian remain
                // retained; takeover failure is owned by the outer cancellation.
                let growth = unsafe { self.attempt_mut()?.arm_runtime(b) }.map_err(helper_error)?;
                self.reserve_growth(growth, b)?;
                State::Armed
            }
            State::Armed => {
                let growth = self
                    .attempt
                    .as_mut()
                    .ok_or_else(|| rejected("missing original compiler attempt"))?
                    .release_runtime(&self.receiver, b)
                    .map_err(helper_error)?;
                self.reserve_growth(growth, b)?;
                State::AwaitingExec
            }
            State::AwaitingExec => {
                if self
                    .attempt_mut()?
                    .poll_first_exec(b)
                    .map_err(helper_error)?
                {
                    State::FirstExec
                } else {
                    State::AwaitingExec
                }
            }
            State::FirstExec => {
                let growth = self
                    .attempt_mut()?
                    .capture_first_exec(b)
                    .map_err(helper_error)?;
                self.reserve_growth(growth, b)?;
                State::HeldExec
            }
            State::HeldExec => {
                // SAFETY: actual original first exec passed image/census checks
                // and remains held. No other FD mutator or creator is admitted.
                unsafe { self.attempt_mut()?.confirm_first_exec(b) }.map_err(helper_error)?;
                State::ConfirmedExec
            }
            State::ConfirmedExec => {
                self.require_cleanup_guard(cleanup, b)?;
                let prepared = self
                    .prepared
                    .take()
                    .ok_or_else(|| rejected("missing original preparation"))?;
                // SAFETY: transfer the exact preparation and confirmed trace into
                // their existing issuer join. The original pool outlives both.
                let growth = unsafe { self.attempt_mut()?.launch_issuer(prepared, cleanup, b) }
                    .map_err(helper_error)?;
                self.reserve_growth(growth, b)?;
                // Issuer creation does not resume the first exec. Join the same
                // retained request while it is still held, before runtime work.
                self.join_original_enrollment(state, b)?;
                State::Running
            }
            State::Running => {
                let enrollment = self
                    .enrollment
                    .as_ref()
                    .ok_or_else(|| rejected("missing original enrollment join"))?;
                let growth = self
                    .attempt
                    .as_mut()
                    .ok_or_else(|| rejected("missing original compiler attempt"))?
                    .service_publication(
                        cleanup,
                        MAX_COMPILER_MODULE_HANDOFF_BYTES_V5,
                        enrollment,
                        b,
                    )
                    .map_err(helper_error)?;
                self.reserve_growth(growth, b)?;
                match self.attempt_mut()?.step_runtime(b).map_err(helper_error)? {
                    Progress::Pending | Progress::Advanced => State::Running,
                    Progress::RootExitHeld => State::RootExitHeld,
                    _ => return Err(rejected("compiler terminated before durable publication")),
                }
            }
            State::RootExitHeld => {
                self.attempt_mut()?
                    .confirm_retired_publication(b)
                    .map_err(helper_error)?;
                State::PublicationObserved
            }
            State::PublicationObserved => {
                self.attempt_mut()?
                    .release_root_exit(b)
                    .map_err(helper_error)?;
                State::AwaitingTerminal
            }
            State::AwaitingTerminal => {
                match self.attempt_mut()?.step_runtime(b).map_err(helper_error)? {
                    Progress::Pending | Progress::Advanced => State::AwaitingTerminal,
                    Progress::RootTerminal { exit_code, signal } => {
                        self.terminal = Some(termination(exit_code, signal)?);
                        State::Retiring
                    }
                    _ => return Err(rejected("compiler lost its released terminal syscall")),
                }
            }
            State::Retiring => match self.attempt_mut()?.step_runtime(b).map_err(helper_error)? {
                Progress::TraceRetired => {
                    let last = self
                        .receiver
                        .last
                        .as_ref()
                        .ok_or_else(|| rejected("missing final original input"))?;
                    let terminal = self
                        .terminal
                        .ok_or_else(|| rejected("missing actual original root wait"))?;
                    let (record, charge) = self
                        .attempt
                        .as_mut()
                        .ok_or_else(|| rejected("missing original compiler attempt"))?
                        .publication_completion(last, b)
                        .map_err(helper_error)?;
                    self.reserve_growth(charge, b)?;
                    if record.terminal().termination() != terminal {
                        return Err(rejected("completion differs from consumed original wait"));
                    }
                    self.completion = Some(record);
                    State::CompletionReady
                }
                _ => {
                    return Err(rejected(
                        "compiler trace retirement did not follow original root wait",
                    ));
                }
            },
            _ => return Err(rejected("compiler request cannot be reused")),
        };
        Ok(self.state == State::CompletionReady)
    }

    pub(crate) fn completion_quota() -> Result<root::CompilerExecutionRootAdmissionQuotaV2> {
        Ok(root::CompilerExecutionRootAdmissionQuotaV2 {
            work: root::sum(&[root::LOCAL_WORK, EXCHANGE_WORK])?,
            scratch: root::sum(&[FRAME, io::packet_receive_scratch(N)])?,
        })
    }

    /// One finite publication attempt after the original orchestration confirms
    /// aggregate shutdown. Every refusal consumes publication; neither cleanup
    /// nor a received description can reconstruct this original record.
    pub(crate) fn publish_after_cleanup(
        &mut self,
        _cleanup: &crate::native_entrypoint::AggregateCleanupComplete,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        self.publish_retired_completion(b)
    }

    pub(crate) fn publish_after_compiler_phase(
        &mut self,
        _phase: &crate::native_entrypoint::fixed_phase::OriginalCompilerPhaseRetired,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        self.publish_retired_completion(b)
    }

    pub(crate) fn require_pending_completion_account(&self, b: &Budget<'_>) -> Result<()> {
        self.check_account(b)?;
        if self.state != State::DrainingCompletion
            || self
                .receiver
                .deadline
                .is_none_or(|deadline| Instant::now() >= deadline)
        {
            return Err(rejected(
                "original compiler completion is no longer pending",
            ));
        }
        Ok(())
    }

    fn publish_retired_completion(&mut self, b: &mut Budget<'_>) -> Result<()> {
        let state = self.state;
        self.state = State::Failed;
        b.charge_work(root::LOCAL_WORK)?;
        self.check_account(b)?;
        if state != State::DrainingCompletion
            || self.prepared.is_some()
            || self.attempt.is_some()
            || self.helper.is_some()
        {
            return Err(rejected("completion requires original terminal cleanup"));
        }
        if self
            .receiver
            .deadline
            .is_none_or(|deadline| Instant::now() >= deadline)
        {
            return Err(rejected("intake deadline exceeded before completion"));
        }
        let connection = self
            .receiver
            .connection
            .as_ref()
            .ok_or_else(|| rejected("missing original connection"))?;
        let peer = net::sockopt::socket_peercred(connection).map_err(|source| Error::Io {
            operation: "original compiler completion peer",
            source,
        })?;
        if self.receiver.sender
            != Some(io::MessageSender::new(
                peer.pid.as_raw_pid(),
                peer.uid.as_raw(),
                peer.gid.as_raw(),
            ))
        {
            return Err(rejected("original compiler completion peer changed"));
        }
        if !self.send_completion(b)? {
            return Err(rejected("original completion publication unavailable"));
        }
        self.state = State::Completed;
        Ok(())
    }

    fn send_completion(&self, b: &mut Budget<'_>) -> Result<bool> {
        b.with_prepaid_scope(
            self.reserved,
            8,
            EXCHANGE_WORK,
            FRAME + io::packet_receive_scratch(N),
            |b| {
                let record = self
                    .completion
                    .as_ref()
                    .ok_or_else(|| rejected("missing original terminal record"))?;
                let last = self
                    .receiver
                    .last
                    .as_ref()
                    .ok_or_else(|| rejected("missing final original input"))?;
                if !record.matches_intake(last, b).map_err(completion_error)? {
                    return Err(rejected("completion differs from original intake"));
                }
                let connection = self
                    .receiver
                    .connection
                    .as_ref()
                    .ok_or_else(|| rejected("missing original connection"))?;
                let sender = self
                    .receiver
                    .sender
                    .ok_or_else(|| rejected("missing original peer"))?;
                if io::receive_authenticated_packet::<N>(connection.as_fd(), sender)
                    .map_err(transport)?
                    .is_some()
                {
                    return Err(rejected("trailing input before completion"));
                }
                Ok(
                    io::send_packet(connection.as_fd(), record.canonical_bytes())
                        .map_err(transport)?
                        .is_some(),
                )
            },
        )
    }
}

fn termination(exit_code: Option<i32>, signal: Option<i32>) -> Result<Termination> {
    match (exit_code, signal) {
        (Some(code @ 0..=255), None) => Ok(Termination::Exited(code as u8)),
        (None, Some(signal @ 1..=64)) => Ok(Termination::Signaled(signal as u8)),
        _ => Err(rejected("noncanonical original terminal wait")),
    }
}

fn completion_error(error: CompletionError) -> Error {
    match error {
        CompletionError::Resource(e) => e.into(),
        CompletionError::Framing(reason) => rejected(reason),
        _ => rejected("original publication completion records differ"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_wait_encoding_requires_exactly_one_canonical_terminal_kind() {
        for code in [0, 1, 101, 255] {
            assert_eq!(
                termination(Some(code), None).unwrap(),
                Termination::Exited(code as u8)
            );
        }
        for signal in [1, 9, 31, 64] {
            assert_eq!(
                termination(None, Some(signal)).unwrap(),
                Termination::Signaled(signal as u8)
            );
        }
        for pair in [
            (None, None),
            (Some(0), Some(9)),
            (Some(-1), None),
            (Some(256), None),
            (None, Some(0)),
            (None, Some(65)),
        ] {
            assert!(termination(pair.0, pair.1).is_err());
        }
    }
}
