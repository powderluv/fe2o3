//! Original attempt transitions and its private installed-confinement receipt.
use super::*;
use crate::native_runtime_guard::Error as GuardError;
use crate::native_v3::NativeAttempt as Issued;
use fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1 as Ledger;
use fe2o3_protected_service_spawn::native_spawn::{
    RootRuntimeTraceV1 as Runtime, RootTaskIdentityV2 as Identity,
    RootTaskObservationV2 as RootView,
};

/// Only the original Attempt's successful gate transition can create this
/// receipt. The trusted pre-READY path installed inherited, irreversible
/// Landlock confinement; the original Domain proves actual pre-clone device
/// denial. A stage flag, syscall result, PID or caller assertion is insufficient.
pub(crate) struct CompilerConfinement {
    identity: Identity,
    ledger: Ledger,
    address: usize,
}

impl CompilerConfinement {
    pub(crate) const STORAGE: usize = size_of::<Self>() + RootView::IDENTITY_STORAGE;
    pub(crate) const VALIDATE_WORK: usize = 8
        + Runtime::IDENTITY_COMPARISON_WORK
        + Runtime::ROOT_OBSERVATION_WORK
        + RootView::VIEW_WORK
        + RootView::DEVICE_CONFINEMENT_WORK;
    pub(crate) const VALIDATE_SCRATCH: usize =
        Self::STORAGE + RootView::VIEW_SCRATCH + RootView::DEVICE_CONFINEMENT_SCRATCH;

    pub(crate) fn validate(
        &self,
        runtime: &Runtime<'_>,
        b: &mut Budget<'_>,
    ) -> std::result::Result<(), GuardError> {
        b.with_prepaid_scope(Self::STORAGE, 8, 8, Self::STORAGE, |b| {
            if self.ledger != b.work_ledger_identity_v1()
                || self.address != b as *const Budget<'_> as usize
                || !runtime.matches_original_identity(&self.identity, b)?
            {
                return Err(GuardError::Invalid(
                    "compiler confinement lost original trace/account identity",
                ));
            }
            runtime.with_task_observation(b, |view, b| Ok(view.require_device_open_confinement(b)?))
        })
    }
}

// These states route the private owner only. Actual trace stops, original Rc
// identity, backing and kernel enforcement remain independently checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Phase {
    Gated,
    Interrupting,
    Interrupted,
    Armed,
    AwaitingExec,
    FirstExec,
    HeldExec,
    ConfirmedExec,
    Issued,
    RootExitHeld,
    PublicationObserved,
    Completed,
    Cancelled,
}

// Issuer retention is prepaid inline; do not introduce allocation during custody transfer.
#[allow(clippy::large_enum_variant)]
pub(super) enum Owner<'work> {
    Gated(Trace<'work>),
    Issued(Issued<'work, Helper>),
}

impl Owner<'_> {
    pub(super) fn cancel(&mut self) -> CleanupPollV1 {
        match self {
            Self::Gated(trace) => trace.cancel(),
            Self::Issued(attempt) => {
                let result = attempt.cancel_compiler();
                let _issuer = attempt.cancel_issuer();
                result
            }
        }
    }

    pub(super) fn needs_foreground_cancellation(&self) -> bool {
        match self {
            Self::Gated(trace) => trace.needs_foreground_cancellation(),
            Self::Issued(attempt) => attempt.needs_foreground_cancellation(),
        }
    }

    pub(super) fn cancel_step(&mut self, b: &mut Budget<'_>) -> AttemptResult<CleanupPollV1> {
        Ok(match self {
            Self::Gated(trace) => trace.cancel_step(b)?,
            Self::Issued(attempt) => attempt.cancel_compiler_step(b)?,
        })
    }
}

impl<'work> Attempt<'work> {
    fn cancel_after_refusal(&mut self) {
        match self.cancel() {
            CleanupPollV1::Pending | CleanupPollV1::Quarantined => {
                // Original custody remains in the owner/pool; refusal does not
                // release it or replace the required foreground/aggregate drain.
            }
            CleanupPollV1::Reaped => {
                // This compiler disposition does not settle the retained issuer,
                // helper or aggregate pool. The same final drain is mandatory.
            }
        }
    }

    pub(super) fn gated_trace(&self) -> AttemptResult<&Trace<'work>> {
        match &self.owner {
            Some(Owner::Gated(trace)) => Ok(trace),
            _ => Err(Failure::Invalid("compiler no longer owns its gated trace")),
        }
    }

    fn gated_trace_mut(&mut self) -> AttemptResult<&mut Trace<'work>> {
        match &mut self.owner {
            Some(Owner::Gated(trace)) => Ok(trace),
            _ => Err(Failure::Invalid("compiler no longer owns its gated trace")),
        }
    }

    /// Interrupt only the same original compiler while its exec gate remains
    /// closed. Neither this state nor a successful interrupt admits execution.
    pub(in super::super) fn interrupt_runtime(
        &mut self,
        received: &Receiver,
        b: &mut Budget<'_>,
    ) -> AttemptResult<()> {
        if self.phase != Phase::Gated || self.controller.is_some() {
            return Err(Failure::Invalid("compiler runtime interrupt is not fresh"));
        }
        self.revalidate(received, b)?;
        self.gated_trace_mut()?.interrupt_for_runtime(b)?;
        self.phase = Phase::Interrupting;
        Ok(())
    }

    /// One consuming original wait; pending does not renew the intake deadline.
    pub(in super::super) fn poll_runtime_interrupt(
        &mut self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<bool> {
        if self.phase != Phase::Interrupting {
            return Err(Failure::Invalid(
                "compiler runtime interrupt was not requested",
            ));
        }
        let event = self.gated_trace_mut()?.poll(b)?;
        if event.is_pending() {
            return Ok(false);
        }
        if !event.is_original_interrupt() {
            self.cancel_after_refusal();
            return Err(Failure::Invalid(
                "compiler gate acquired an unexpected original stop",
            ));
        }
        self.phase = Phase::Interrupted;
        Ok(true)
    }

    pub(in super::super) fn arm_runtime_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let inner = Trace::runtime_takeover_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, inner.work()])?,
            scratch: native::sum(&[FRAME, inner.scratch()])?,
        })
    }

    /// Consume the original held interrupt and retain the new runtime owner
    /// before any later fallible accounting. Returned growth is unreserved.
    ///
    /// # Safety
    /// Preserve the dedicated process/outside custodian and original exclusive
    /// creator/account contracts through full foreground retirement. The original
    /// gate stays closed; an unresolved failure after takeover must fail-stop,
    /// never transfer an armed tree to root-only background cleanup.
    #[allow(unsafe_code)]
    pub(in super::super) unsafe fn arm_runtime(
        &mut self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<usize> {
        if self.phase != Phase::Interrupted || self.controller.is_some() {
            return Err(Failure::Invalid(
                "compiler runtime takeover lacks its held interrupt",
            ));
        }
        let retained = native::sum(&[self.retained, Runtime::STORAGE_GROWTH])?;
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            let Some(Owner::Gated(trace)) = self.owner.take() else {
                return Err(Failure::Invalid(
                    "compiler runtime takeover lost original owner",
                ));
            };
            self.phase = Phase::Cancelled;
            // SAFETY: the caller preserves the original closed gate and dedicated
            // custody. This consumes only the trace held in this same Attempt.
            let (trace, _) = unsafe { trace.arm_runtime(b) }?;
            self.owner = Some(Owner::Gated(trace));
            b.reserve_storage(Runtime::STORAGE_GROWTH)?;
            self.retained = retained;
            self.phase = Phase::Armed;
            Ok(Runtime::STORAGE_GROWTH)
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Revalidate the original received/staged objects, establish actual device
    /// confinement and release only its held bootstrap interrupt. The trusted
    /// bootstrap installs the additional TRACE filter before original exec;
    /// no application instruction is resumed by this transition.
    pub(in super::super) fn release_runtime(
        &mut self,
        received: &Receiver,
        b: &mut Budget<'_>,
    ) -> AttemptResult<usize> {
        let result = (|| {
            if self.phase != Phase::Armed || self.controller.is_some() || self.confinement.is_some()
            {
                return Err(Failure::Invalid(
                    "compiler runtime gate is not freshly armed",
                ));
            }
            self.revalidate(received, b)?;
            let retained = native::sum(&[self.retained, CompilerConfinement::STORAGE])?;
            b.with_prepaid_scope(
                self.retained,
                8,
                LOCAL_WORK,
                FRAME + CompilerConfinement::STORAGE,
                |b| {
                    require_deadline(self.deadline)?;
                    let gate = self
                        .gate_writer
                        .as_ref()
                        .ok_or(Failure::Invalid("compiler runtime gate writer is absent"))?;
                    let Some(Owner::Gated(trace)) = &mut self.owner else {
                        return Err(Failure::Invalid("compiler gate lost original owner"));
                    };
                    let confinement = trace.with_runtime_backing(b, |runtime, _, b| {
                        let event = runtime.poll(b)?.ok_or(Failure::Invalid(
                            "compiler gate lacks its held bootstrap interrupt",
                        ))?;
                        if !event.is_interrupt() || event.pid() != runtime.pid() {
                            return Err(Failure::Invalid(
                                "compiler gate is not at its original bootstrap interrupt",
                            ));
                        }
                        let (identity, storage) = runtime.with_task_observation(b, |view, b| {
                            view.require_device_open_confinement(b)?;
                            view.retain_identity(b)
                        })?;
                        b.reserve_storage(storage.additional_storage())?;
                        launch_io::release_child(
                            gate.as_fd(),
                            &mut GateObserver { runtime, budget: b },
                            self.deadline,
                        )?;
                        runtime.resume_selected(b)?;
                        Ok::<_, Failure>(CompilerConfinement {
                            identity,
                            ledger: b.work_ledger_identity_v1(),
                            address: b as *const Budget<'_> as usize,
                        })
                    })?;
                    self.confinement = Some(confinement);
                    b.reserve_storage(CompilerConfinement::STORAGE)?;
                    self.retained = retained;
                    drop(self.gate_writer.take());
                    self.phase = Phase::AwaitingExec;
                    Ok(CompilerConfinement::STORAGE)
                },
            )
        })();
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Only Pending or the exact original root Exec may follow gate release.
    /// Keep that Exec held through image validation and issuer readiness.
    pub(in super::super) fn poll_first_exec(&mut self, b: &mut Budget<'_>) -> AttemptResult<bool> {
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            if self.phase != Phase::AwaitingExec || self.confinement.is_none() {
                return Err(Failure::Invalid(
                    "compiler first-exec poll has no released gate",
                ));
            }
            require_deadline(self.deadline)?;
            let found = self
                .gated_trace_mut()?
                .with_runtime_backing(b, |runtime, _, b| {
                    let Some(event) = runtime.poll(b)? else {
                        return Ok::<_, Failure>(false);
                    };
                    if !event.is_exec() || event.pid() != runtime.pid() {
                        return Err(Failure::Invalid(
                            "compiler bootstrap did not reach its original first exec",
                        ));
                    }
                    Ok(true)
                })?;
            if found {
                self.phase = Phase::FirstExec;
            }
            Ok(found)
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// First-exec policy capture only. The gate/exec protocol remains a separate
    /// transition; this refuses unless the actual original runtime holds Exec.
    pub(in super::super) fn capture_first_exec(
        &mut self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<usize> {
        if self.phase != Phase::FirstExec || self.controller.is_some() {
            return Err(Failure::Invalid("compiler first-exec capture is not fresh"));
        }
        let retained = native::sum(&[self.retained, Controller::STORAGE])?;
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            let Some(Owner::Gated(trace)) = &mut self.owner else {
                return Err(Failure::Invalid("compiler first-exec lost original owner"));
            };
            let inventory = &self.executables;
            let confinement = self.confinement.take().ok_or(Failure::Invalid(
                "compiler first-exec lacks its original confinement receipt",
            ))?;
            let controller = trace.with_runtime_backing(b, |runtime, helper, b| {
                helper.with_compiler(b, |backing, b| -> AttemptResult<_> {
                    inventory.revalidate(backing, b)?;
                    Ok(Controller::from_confined_root_exec(
                        runtime,
                        backing,
                        inventory,
                        confinement,
                        b,
                    )?)
                })
            })?;
            self.controller = Some(controller);
            b.reserve_storage(Controller::STORAGE)?;
            self.retained = retained;
            self.phase = Phase::HeldExec;
            Ok(Controller::STORAGE)
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Close every original staging/gate alias before checking native exec EOF.
    /// Neither EOF nor confirmation grants permission to run application code.
    ///
    /// # Safety
    /// The dedicated original custodian excludes fork/FD mutation outside this
    /// trace. All first-exec/profile/cgroup obligations of the original launch
    /// remain held; this actual stop has already passed complete image/census
    /// validation and has not resumed since it was captured.
    #[allow(unsafe_code)]
    pub(in super::super) unsafe fn confirm_first_exec(
        &mut self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<()> {
        if self.phase != Phase::HeldExec || self.controller.is_none() {
            return Err(Failure::Invalid(
                "compiler exec confirmation lacks held image validation",
            ));
        }
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            require_deadline(self.deadline)?;
            // Stage owns duplicate status/gate/artifact sources, so it must die
            // before EOF. Retain all prepaid charges conservatively until Drop.
            drop(self.stage.take());
            drop(self.gate_reader.take());
            drop(self.gate_writer.take());
            let mut status = [0_u8; 1];
            if rustix::io::read(&self._exec_reader, &mut status)
                .map_err(|e| native::io("read original compiler exec status", e))?
                != 0
            {
                return Err(Failure::Invalid(
                    "compiler exec status did not reach clean EOF",
                ));
            }
            // SAFETY: all parent aliases closed above. Original held Exec and
            // image/census passed without resume; the caller excludes escapes.
            unsafe { self.gated_trace_mut()?.confirm_exec(b) }?;
            self.phase = Phase::ConfirmedExec;
            Ok(())
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Move the exact original trace into the existing issuer composition. The
    /// controller and inventory never leave this Attempt or get reconstructed.
    /// Returned growth is above this Attempt PLUS the consumed preparation.
    ///
    /// # Safety
    /// Preserve Prepared::launch_root_attempt's original creator, cleanup and
    /// payload Drop contracts. The existing independently funded original pool
    /// must outlive both compiler and issuer, including unresolved fail-stop.
    #[allow(unsafe_code)]
    pub(in super::super) unsafe fn launch_issuer(
        &mut self,
        prepared: crate::native_v3::PreparedCompilerExecutionSupervisorV3,
        cleanup: &mut Cleanup,
        b: &mut Budget<'work>,
    ) -> AttemptResult<usize> {
        if self.phase != Phase::ConfirmedExec || self.controller.is_none() {
            return Err(Failure::Invalid(
                "issuer requires original confirmed compiler image",
            ));
        }
        let input = native::sum(&[self.retained, prepared.retained_storage()])?;
        let result = b.with_prepaid_scope(input, 8, LOCAL_WORK, FRAME, |b| {
            require_deadline(self.deadline)?;
            let Some(Owner::Gated(trace)) = self.owner.take() else {
                return Err(Failure::Invalid(
                    "issuer transfer lost original compiler trace",
                ));
            };
            self.phase = Phase::Cancelled;
            // SAFETY: the caller retains the original dedicated cleanup contract.
            // This is the confirmed original trace, not independently supplied IDs.
            let (attempt, growth) = unsafe {
                prepared.launch_root_attempt(
                    trace,
                    self.deadline.saturating_duration_since(Instant::now()),
                    cleanup,
                    b,
                )
            }?;
            self.owner = Some(Owner::Issued(attempt));
            b.reserve_storage(growth.additional_storage())?;
            let retained = native::sum(&[input, growth.additional_storage()])?;
            self.retained = retained;
            let Some(Owner::Issued(attempt)) = &self.owner else {
                unreachable!("retained original issuer")
            };
            attempt.validate_ready(b)?;
            require_deadline(self.deadline)?;
            self.phase = Phase::Issued;
            Ok(growth.additional_storage())
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Only an actual issuer-owning variant may run checkpoints. Readiness is
    /// checked while the first exec stays held; subsequent steps keep the same
    /// original helper/inventory/controller and do not hash code per syscall.
    pub(in super::super) fn step_runtime(
        &mut self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<crate::native_runtime_controller::Progress> {
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            if !matches!(self.phase, Phase::Issued | Phase::RootExitHeld) {
                return Err(Failure::Invalid(
                    "compiler checkpoint lacks original issued attempt",
                ));
            }
            let Some(Owner::Issued(attempt)) = &mut self.owner else {
                return Err(Failure::Invalid("compiler issuer custody is absent"));
            };
            let controller = self
                .controller
                .as_mut()
                .ok_or(Failure::Invalid("compiler controller is absent"))?;
            let inventory = &self.executables;
            let progress = attempt.with_runtime_backing(b, |runtime, helper, b| {
                helper.with_compiler_checkpoint(b, |backing, b| -> AttemptResult<_> {
                    Ok(controller.step(runtime, backing, inventory, b)?)
                })
            })?;
            if progress == crate::native_runtime_controller::Progress::RootExitHeld {
                self.phase = Phase::RootExitHeld;
            }
            Ok(progress)
        });
        if result.is_err() {
            // This wrapper owns the genuine trace; no supplied runtime can be
            // cancelled. The controller's standalone wrong-owner refusal stays
            // non-mutating for an unrelated runtime passed to that lower API.
            self.cancel_after_refusal();
        }
        result
    }

    /// Pump the original issuer while rustc waits for Prepare/Issue/Publish.
    /// This shares the request's original deadline, account and runtime owner.
    pub(in super::super) fn service_publication(
        &mut self,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        enrollment: &Option<fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1>,
        b: &mut Budget<'_>,
    ) -> AttemptResult<usize> {
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            require_deadline(self.deadline)?;
            if self.phase != Phase::Issued || self.controller.is_none() {
                return Err(Failure::Invalid(
                    "root RPC lacks original running controller",
                ));
            }
            let Some(Owner::Issued(attempt)) = &mut self.owner else {
                return Err(Failure::Invalid("root RPC lost original issued owner"));
            };
            let growth =
                attempt.service_publication(cleanup, maximum_handoff_bytes, enrollment, b)?;
            b.reserve_storage(growth.additional_storage())?;
            self.retained = native::sum(&[self.retained, growth.additional_storage()])?;
            require_deadline(self.deadline)?;
            Ok(growth.additional_storage())
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Require the original durable retirement before permitting terminal exit.
    /// No second late holder or post-retirement publication lock is acquired.
    pub(in super::super) fn confirm_retired_publication(
        &mut self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<()> {
        let result = (|| {
            if self.phase != Phase::RootExitHeld {
                return Err(Failure::Invalid(
                    "retirement requires original held root exit",
                ));
            }
            if self.step_runtime(b)? != crate::native_runtime_controller::Progress::RootExitHeld {
                return Err(Failure::Invalid("publication lost original held root exit"));
            }
            b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
                let Some(Owner::Issued(attempt)) = &mut self.owner else {
                    return Err(Failure::Invalid("publication lost original issued owner"));
                };
                attempt.require_retired_publication(b)?;
                self.phase = Phase::PublicationObserved;
                Ok(())
            })
        })();
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    /// Recheck the original durable tombstone while its original root remains
    /// held, then let only that exact terminal syscall proceed. This returns no
    /// completion or publication authority; consuming terminal waits are next.
    pub(in super::super) fn release_root_exit(&mut self, b: &mut Budget<'_>) -> AttemptResult<()> {
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            if self.phase != Phase::PublicationObserved {
                return Err(Failure::Invalid(
                    "root exit lacks original publication custody",
                ));
            }
            let Some(Owner::Issued(attempt)) = &mut self.owner else {
                return Err(Failure::Invalid("root exit lost original issued owner"));
            };
            attempt.require_retired_publication(b)?;
            let controller = self
                .controller
                .as_mut()
                .ok_or(Failure::Invalid("root exit lost original controller"))?;
            let inventory = &self.executables;
            attempt.with_runtime_backing(b, |runtime, helper, b| {
                helper.with_compiler_checkpoint(b, |backing, b| -> AttemptResult<_> {
                    Ok(controller.resume_root_exit(runtime, backing, inventory, b)?)
                })
            })?;
            self.phase = Phase::Issued;
            Ok(())
        });
        if result.is_err() {
            self.cancel_after_refusal();
        }
        result
    }

    pub(in super::super) fn publication_observation_quota(
        maximum_handoff_bytes: usize,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let held = Self::runtime_step_quota()?;
        let publication = Issued::<Helper>::publication_observation_quota(maximum_handoff_bytes)?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[held.work(), LOCAL_WORK, publication.work()])?,
            scratch: native::sum(&[held.scratch(), FRAME, publication.scratch()])?,
        })
    }

    pub(in super::super) fn publication_service_quota(
        maximum_handoff_bytes: usize,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let rpc = Issued::<Helper>::publication_service_quota(maximum_handoff_bytes)?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, rpc.work()])?,
            scratch: native::sum(&[FRAME, rpc.scratch()])?,
        })
    }

    pub(in super::super) fn retired_publication_confirmation_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let step = Self::runtime_step_quota()?;
        let retired = Issued::<Helper>::retired_publication_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[step.work(), LOCAL_WORK, retired.work()])?,
            scratch: native::sum(&[step.scratch(), FRAME, retired.scratch()])?,
        })
    }

    pub(in super::super) fn root_exit_release_quota(
        &self,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let Some(Owner::Issued(_)) = &self.owner else {
            return Err(Failure::Invalid("root exit has no original issued owner"));
        };
        let publication = Issued::<Helper>::retired_publication_quota()?;
        // The same fixed step ceiling conservatively funds the exact held-entry
        // recheck, fresh census, mapping validation and selected syscall step.
        let step = Self::runtime_step_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[step.work(), publication.work()])?,
            scratch: native::sum(&[step.scratch(), publication.scratch()])?,
        })
    }

    pub(in super::super) fn runtime_step_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let trace = Issued::<Helper>::runtime_backing_quota()?;
        let helper = Helper::checkpoint_access_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[
                LOCAL_WORK,
                trace.work(),
                helper.work(),
                Controller::step_work()?,
            ])?,
            scratch: native::sum(&[
                FRAME,
                trace.scratch(),
                helper.scratch(),
                Controller::STEP_SCRATCH,
            ])?,
        })
    }

    /// Copy only the original issued owner's actual retired publication result.
    /// The record is inert; the request's authenticated original channel and
    /// parent's retained publication join remain separate obligations.
    pub(in super::super) fn publication_completion(
        &mut self,
        last: &fe2o3_compiler_execution_protocol::CompilerExecutionRootIntakeRecordV4,
        b: &mut Budget<'_>,
    ) -> AttemptResult<(
        fe2o3_compiler_execution_protocol::CompilerExecutionRootPublicationCompletionV1,
        usize,
    )> {
        let result = b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            if self.phase != Phase::Issued || self.controller.is_none() {
                return Err(Failure::Invalid(
                    "completion lost original issued controller",
                ));
            }
            let Some(Owner::Issued(attempt)) = &mut self.owner else {
                return Err(Failure::Invalid("completion lost original issued owner"));
            };
            let (record, charge) = attempt.publication_completion(last, b)?;
            Ok((record, charge.additional_storage()))
        });
        match result {
            Ok(value) => {
                self.phase = Phase::Completed;
                Ok(value)
            }
            Err(error) => {
                self.cancel_after_refusal();
                Err(error)
            }
        }
    }

    pub(in super::super) fn publication_completion_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let inner = Issued::<Helper>::publication_completion_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, inner.work()])?,
            scratch: native::sum(&[FRAME, inner.scratch()])?,
        })
    }

    pub(in super::super) fn runtime_poll_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        Trace::gated_operation_quota().map_err(Into::into)
    }

    pub(in super::super) fn runtime_confirmation_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let inner = Trace::runtime_confirmation_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            // LOCAL_WORK includes the one bounded native status read and alias closures.
            work: native::sum(&[LOCAL_WORK, inner.work()])?,
            scratch: native::sum(&[FRAME, inner.scratch()])?,
        })
    }

    pub(in super::super) fn runtime_issuer_quota(
        &self,
        prepared: &crate::native_v3::PreparedCompilerExecutionSupervisorV3,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let launch = prepared.issuer_launch_quota(self.gated_trace()?)?;
        let ready = prepared.issuer_continuity_quota::<Helper>()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, launch.work(), ready.work()])?,
            scratch: native::sum(&[FRAME, launch.scratch(), ready.scratch()])?,
        })
    }

    pub(in super::super) fn maximum_runtime_issuer_quota(
        compiler_payload: usize,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        use crate::native_v3::PreparedCompilerExecutionSupervisorV3 as Prepared;
        let launch = Prepared::maximum_issuer_launch_quota::<Helper>(compiler_payload)?;
        let ready = Prepared::maximum_issuer_continuity_quota::<Helper>()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, launch.work(), ready.work()])?,
            scratch: native::sum(&[FRAME, launch.scratch(), ready.scratch()])?,
        })
    }

    pub(in super::super) fn maximum_root_exit_release_quota(
        _maximum_handoff_bytes: usize,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let publication = Issued::<Helper>::retired_publication_quota()?;
        let step = Self::runtime_step_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[step.work(), publication.work()])?,
            scratch: native::sum(&[step.scratch(), publication.scratch()])?,
        })
    }

    pub(in super::super) fn runtime_gate_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let trace = Trace::runtime_backing_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            // Full received/stage validation is quoted separately by request.
            work: native::sum(&[
                LOCAL_WORK,
                trace.work(),
                2 * Runtime::OPERATION_WORK,
                Runtime::ROOT_OBSERVATION_WORK,
                RootView::VIEW_WORK,
                RootView::DEVICE_CONFINEMENT_WORK,
                RootView::IDENTITY_WORK,
                launch_io::MAX_GATE_ATTEMPTS * launch_io::Boundary::GateRelease.work(),
            ])?,
            scratch: native::sum(&[
                FRAME,
                CompilerConfinement::STORAGE,
                trace.scratch(),
                Runtime::OPERATION_SCRATCH,
                RootView::VIEW_SCRATCH,
                RootView::DEVICE_CONFINEMENT_SCRATCH,
                RootView::IDENTITY_SCRATCH,
                RootView::IDENTITY_STORAGE,
            ])?,
        })
    }

    pub(in super::super) fn first_exec_poll_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let trace = Trace::runtime_backing_quota()?;
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, trace.work(), Runtime::OPERATION_WORK])?,
            scratch: native::sum(&[FRAME, trace.scratch(), Runtime::OPERATION_SCRATCH])?,
        })
    }

    pub(in super::super) fn continuity_quota(
        &self,
    ) -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        if !matches!(self.owner, Some(Owner::Issued(_))) {
            return Err(Failure::Invalid(
                "compiler has no original issued continuity",
            ));
        }
        Self::maximum_continuity_quota()
    }

    pub(in super::super) fn maximum_continuity_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let inner = Issued::<Helper>::original_validation_quota();
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, inner.work()])?,
            scratch: native::sum(&[FRAME, inner.scratch()])?,
        })
    }

    pub(in super::super) fn continuity(&self, b: &mut Budget<'_>) -> AttemptResult<()> {
        b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            let Some(Owner::Issued(attempt)) = &self.owner else {
                return Err(Failure::Invalid(
                    "compiler has no original issued continuity",
                ));
            };
            Ok(attempt.validate_original(b)?)
        })
    }

    /// Scoped descriptive query through this original issued owner. The caller
    /// uses the closed request enrollment quota, plus its callback's own costs.
    /// No execution transition, transport record or mapped recovery is enabled.
    pub(in super::super) fn with_original_enrollment<R>(
        &self,
        b: &mut Budget<'_>,
        operation: impl FnOnce(
            &crate::native_v3::OriginalCompilerEnrollment,
            &mut Budget<'_>,
        ) -> AttemptResult<R>,
    ) -> AttemptResult<R> {
        b.check_prior_denials_v1()?;
        b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            if self.phase != Phase::Issued || self.controller.is_none() {
                return Err(Failure::Invalid(
                    "enrollment lost original issued controller",
                ));
            }
            let Some(Owner::Issued(attempt)) = &self.owner else {
                return Err(Failure::Invalid("enrollment lost original issued owner"));
            };
            attempt.with_original_enrollment(b, operation)
        })
    }

    pub(in super::super) fn original_policy_identity_quota()
    -> AttemptResult<native::CompilerExecutionLaunchQuotaV2> {
        let inner = Issued::<Helper>::original_policy_identity_quota();
        Ok(native::CompilerExecutionLaunchQuotaV2 {
            work: native::sum(&[LOCAL_WORK, inner.work()])?,
            scratch: native::sum(&[FRAME, inner.scratch()])?,
        })
    }

    pub(in super::super) fn original_policy_identity(
        &self,
        b: &mut Budget<'_>,
    ) -> AttemptResult<[u8; 32]> {
        b.with_prepaid_scope(self.retained, 8, LOCAL_WORK, FRAME, |b| {
            let Some(Owner::Issued(attempt)) = &self.owner else {
                return Err(Failure::Invalid("compiler has no original issued policy"));
            };
            Ok(attempt.original_policy_identity(b)?)
        })
    }
}

struct GateObserver<'a, 'work, 'budget> {
    runtime: &'a Runtime<'work>,
    budget: &'a mut Budget<'budget>,
}
impl launch_io::Observer for GateObserver<'_, '_, '_> {
    type Error = Failure;
    fn before_attempt(&mut self, boundary: launch_io::Boundary) -> AttemptResult<()> {
        Ok(self.budget.charge_work(boundary.work())?)
    }
    fn is_live(&mut self) -> AttemptResult<bool> {
        self.runtime.with_task_observation(self.budget, |view, b| {
            view.validate_continuity(b)?;
            Ok(true)
        })
    }
}
