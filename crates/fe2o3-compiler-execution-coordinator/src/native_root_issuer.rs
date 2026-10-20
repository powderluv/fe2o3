// Direct root-to-issuer composition. No handoff, deployment or occurrence authority
// is manufactured here. The attempt owns its original trace/session independently
// of the issuer; runtime admission and publication remain separate integrations.
use super::PreparedCompilerExecutionSupervisorV3 as Prepared;
use crate::compiler_child_channel::{CompilerTrace, CompilerTraceEvent as TraceEvent};
use crate::native_launch::{
    self as launch, CompilerExecutionLaunchErrorV2 as Error,
    CompilerExecutionLaunchQuotaV2 as Quota, CompilerExecutionLaunchStorageV2 as Storage,
    FILE_STORAGE, Observer, Result, sum,
};
use fe2o3_broker_authority_service::{
    RootConnectionV3 as RootConnection, RootControlSessionV3 as RootSession,
    RootLaunchChannelV3 as RootChannel, RootPublicationCustodyV3 as Publication,
};
use fe2o3_compiler_closure_capability::{
    CompilerExecutionPolicyCapabilityV3 as PolicyCap,
    CompilerExecutionServiceLaunchCapabilityV3 as ManifestCap,
    CompilerExecutionSigningKeyServiceTransferV3 as ServiceKey,
};
use fe2o3_compiler_execution_protocol::{
    COMPILER_EXECUTION_SERVICE_LAUNCH_MANIFEST_STORAGE_V3 as MANIFEST_SCRATCH,
    COMPILER_EXECUTION_SERVICE_LAUNCH_MANIFEST_WORK_V3 as MANIFEST_WORK,
    COMPILER_EXECUTION_SERVICE_READY_BYTES_V3 as READY_BYTES,
    COMPILER_EXECUTION_SERVICE_READY_STORAGE_V3 as READY_SCRATCH,
    COMPILER_EXECUTION_SERVICE_READY_WORK_V3 as READY_WORK,
    CompilerExecutionAttestationStorageV3 as RecordStorage,
    CompilerExecutionClientProcessIdentityV1 as Client,
    CompilerExecutionServiceLaunchManifestV3 as Manifest, CompilerExecutionServiceReadyV3 as Ready,
};
use fe2o3_compiler_execution_supervisor::ProvisionedProtectedIssuerServiceInputsV2 as Inputs;
use fe2o3_external_anchor_coordinator::ExternalAnchorSupervisorTransferV3 as AnchorTransfer;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
    CanonicalKernelIrWorkLedgerIdentityV1 as Ledger,
};
use fe2o3_protected_service_spawn::{
    ProtectedServiceCleanupServiceV2 as Cleanup, ProtectedServiceDescriptorBindingV1 as Binding,
    RetainedDependencyV2 as Dependency, RetainedResourcesV2 as Resources,
    cleanup_bridge::CleanupPollV1 as CleanupPoll,
    launch_io,
    native_spawn::{
        RootOwnedProtectedServiceChildV2 as PlainChild,
        RootOwnedRetainedServiceChildV2 as RetainedChild, StagedProtectedServiceExecV2 as Stage,
    },
};
use fe2o3_protected_static_executable::{
    ProtectedStaticExecutableOperationV2 as ImageOperation, ProtectedStaticExecutableV2 as Image,
};
pub(crate) use quota::IssuerCleanupQuota;
use rustix::{event, fs, io, net, pipe};
use staging::{Channels, Readers};
use std::{
    fmt,
    marker::PhantomData,
    mem::size_of,
    os::fd::{AsFd, BorrowedFd, OwnedFd},
    rc::Rc,
    time::{Duration, Instant},
};

type Child<T> = RetainedChild<Payload<T>>;
const READY_OWNER: usize = size_of::<(Ready, RecordStorage)>();
const MANIFEST_OWNER: usize = size_of::<(Manifest, RecordStorage)>();
// Fixed table/metadata checks, descriptor creation/closure, and refusal cleanup.
// Every native image, record, retained access and transport charges separately.
const LOCAL_WORK: usize = 8 + 128 * 1088;
const FRAME: usize = 4 * size_of::<(NativeAttempt<'static, ()>, Storage)>()
    + 4 * size_of::<Stage>()
    + 8 * size_of::<Error>()
    + 4 * size_of::<fs::Stat>()
    + 16384;

struct Payload<T: Send + 'static> {
    // The unused supervisor and launcher, all leases and live anchor stay owned.
    prepared: Prepared,
    _dependency: Dependency<T>,
    key: ServiceKey,
    manifest: ManifestCap,
    retained: usize,
}

// Policy-neutral staging view over actual retained owners. Compiler and
// application payloads remain distinct; this view cannot create either one.
struct StageInputs<'a> {
    prepared: &'a Prepared,
    key: &'a ServiceKey,
    manifest: &'a ManifestCap,
    retained: usize,
}
impl<T: Send + 'static> Payload<T> {
    const ENVELOPE: usize = size_of::<(Self, usize)>()
        - size_of::<Prepared>()
        - size_of::<Dependency<T>>()
        - size_of::<ServiceKey>()
        - size_of::<ManifestCap>();
}

// Accounting identity only, meaningful while the original Work borrow remains
// live. Captured in the authenticated trace callback, never from decoded input.
struct RequestAccount {
    ledger: Ledger,
    address: usize,
}
impl RequestAccount {
    fn capture(b: &Budget<'_>) -> Self {
        Self {
            ledger: b.work_ledger_identity_v1(),
            address: b as *const Budget<'_> as usize,
        }
    }

    fn with<R>(
        &self,
        retained: usize,
        b: &mut Budget<'_>,
        operation: impl FnOnce(&mut Budget<'_>) -> Result<R>,
    ) -> Result<R> {
        b.with_prepaid_scope(retained, 8, LOCAL_WORK, FRAME, |b| {
            if self.ledger != b.work_ledger_identity_v1()
                || self.address != b as *const Budget<'_> as usize
            {
                return Err(Resource::Accounting.into());
            }
            operation(b)
        })
    }
}

/// Original compiler/session custody, independent of the removable issuer.
/// This owner cannot resume the compiler or grant publication/launch authority.
/// Keep the creator thread and original cleanup controller alive until termination,
/// including deferred/quarantined cleanup. Retire the FULL charge only after Drop.
/// Removing the issuer does not reduce this charge or replace the session/trace.
/// Keep the original Work borrow live and Budget at its admitting address until
/// this owner drops; address equality is not persistent identity after that borrow.
pub(crate) struct NativeAttempt<'work, T: Send + 'static> {
    // Field order starts compiler cancellation before dropping either other owner.
    trace: CompilerTrace<'work, T>,
    root: RootSession<'work>,
    issuer: Option<ManagedIssuer<'work, T>>,
    publication: Option<Publication>,
    retained: usize,
    continuity: Quota,
    account: RequestAccount,
    // Never send the foreground launch owner or imply a Send escape for the trace.
    _creator: PhantomData<(&'work Budget<'work>, Rc<()>)>,
}
impl<'work, T: Send + 'static> NativeAttempt<'work, T> {
    const ENVELOPE: usize = size_of::<(Self, Storage)>()
        - size_of::<CompilerTrace<'static, T>>()
        - size_of::<RootSession<'static>>()
        - size_of::<ManagedIssuer<'static, T>>();
    pub(crate) fn issuer_pid(&self) -> Option<rustix::process::Pid> {
        self.issuer.as_ref().map(|issuer| issuer.child.pid())
    }
    pub(crate) fn readiness(&self) -> Option<&Ready> {
        self.issuer.as_ref().map(|issuer| &issuer.ready)
    }
    pub(crate) const fn retained_storage(&self) -> usize {
        self.retained
    }
    pub(crate) const fn continuity_quota(&self) -> Quota {
        self.continuity
    }

    /// Read the policy association from the original retained preparation after
    /// its transfer. This does not replace live continuity or grant authority.
    pub(crate) fn original_policy_identity(&self, b: &mut Budget<'_>) -> Result<[u8; 32]> {
        self.original_policy_coordinates(b).map(|(identity, _)| identity)
    }

    fn original_policy_coordinates(&self, b: &mut Budget<'_>) -> Result<([u8; 32], u64)> {
        self.account.with(self.retained, b, |b| {
            self.issuer
                .as_ref()
                .ok_or(Error::Invalid(
                    "root attempt has no original issuer payload",
                ))?
                .child
                .with_resources(b, |payload, _| {
                    let policy = payload.prepared.trust.policy().policy();
                    Ok((*policy.identity().as_bytes(), policy.generation()))
                })
        })
    }

    /// Owning attempt must check readiness/continuity before resuming its compiler.
    /// This does not perform Prepare/Issue or inspect a not-yet-created publication.
    pub(crate) fn validate_ready(&self, b: &mut Budget<'_>) -> Result<()> {
        self.account.with(self.retained, b, |b| {
            let issuer = self
                .issuer
                .as_ref()
                .ok_or(Error::Invalid("root attempt has no issuer"))?;
            issuer.child.with_resources(b, |p, b| -> Result<()> {
                p.prepared.validate_process(issuer.child.pid(), b)?;
                match_ready(
                    &issuer.ready,
                    issuer.child.pid(),
                    &p.manifest,
                    &p.prepared,
                    b,
                )?;
                // Connection validation includes the actual running image on
                // this same retained child and policy, after the root join.
                self.trace
                    .with_observation(b, |original, b| -> Result<()> {
                        Ok(issuer.connection.validate(
                            &self.root,
                            original,
                            &issuer.child,
                            p.prepared.trust.policy().policy(),
                            p.manifest.manifest(),
                            b,
                        )?)
                    })?;
                Ok(())
            })?;
            require_live(&issuer.child, b)
        })
    }

    /// Revalidate only the original live compiler/session, never issuer readiness.
    pub(crate) fn validate_original(&self, b: &mut Budget<'_>) -> Result<()> {
        self.account.with(self.retained, b, |b| {
            self.trace.with_observation(b, |original, b| -> Result<()> {
                Ok(self.root.validate_original(original, b)?)
            })
        })
    }

    /// Attach actual publication custody to this original attempt independently
    /// of its issuer. Any error or unwind consumes/cancels the attempt, including
    /// failure of the OUTERMOST accounting scope after successful acquisition.
    /// The original cleanup slot retains partial custody until terminal cleanup.
    /// Returned growth is unreserved above the complete consumed attempt charge.
    pub(crate) fn observe_publication(
        mut self,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        b: &mut Budget<'_>,
    ) -> Result<(Self, Storage)> {
        let (publication, retained, growth) = self.account.with(self.retained, b, |b| {
            if self.publication.is_some() {
                return Err(Error::Invalid("root attempt already owns a publication"));
            }
            self.trace
                .with_observation(b, |original, b| -> Result<()> {
                    Ok(self.root.validate_original(original, b)?)
                })?;
            let (publication, growth) =
                self.trace
                    .observe_publication(cleanup, maximum_handoff_bytes, b)?;
            b.reserve_storage(growth)?;
            self.trace
                .with_observation(b, |original, b| -> Result<()> {
                    Ok(self.root.validate_original(original, b)?)
                })?;
            let retained = sum(&[self.retained, growth])?;
            Ok((publication, retained, Storage(growth)))
        })?;
        // Only infallible moves remain after the complete outer transaction.
        self.publication = Some(publication);
        self.retained = retained;
        Ok((self, growth))
    }

    pub(crate) fn revalidate_publication(&self, b: &mut Budget<'_>) -> Result<()> {
        self.account.with(self.retained, b, |b| {
            let publication = self
                .publication
                .as_ref()
                .ok_or(Error::Invalid("root attempt has no publication"))?;
            self.trace
                .with_observation(b, |original, b| -> Result<()> {
                    Ok(self.root.validate_original(original, b)?)
                })?;
            self.trace.revalidate_publication(publication, b)?;
            self.trace.with_observation(b, |original, b| -> Result<()> {
                Ok(self.root.validate_original(original, b)?)
            })
        })
    }

    /// Original consuming wait only; neither readiness nor runtime admission.
    pub(crate) fn poll_compiler(&mut self, b: &mut Budget<'_>) -> Result<TraceEvent> {
        b.with_prepaid_scope(self.retained, 0, 0, 0, |b| self.trace.poll(b))
    }

    pub(crate) fn cancel_compiler(&mut self) -> CleanupPoll {
        self.trace.cancel()
    }

    /// Advance the same compiler's foreground runtime cancellation. A Pending
    /// result retains the whole attempt; it is not a background cleanup transfer.
    pub(crate) fn cancel_compiler_step(&mut self, b: &mut Budget<'_>) -> Result<CleanupPoll> {
        self.account
            .with(self.retained, b, |b| self.trace.cancel_step(b))
    }

    pub(crate) fn needs_foreground_cancellation(&self) -> bool {
        self.trace.needs_foreground_cancellation()
    }

    /// The same retained compiler and payload, without exposing an old mutable
    /// trace or a PID-based reconstruction. Callback policy/retention is additional;
    /// issuer readiness and publication remain separate checked transitions.
    pub(crate) fn with_runtime_backing<R, E>(
        &mut self,
        b: &mut Budget<'_>,
        operation: impl FnOnce(
            &mut fe2o3_protected_service_spawn::native_spawn::RootRuntimeTraceV1<'work>,
            &T,
            &mut Budget<'_>,
        ) -> std::result::Result<R, E>,
    ) -> std::result::Result<R, E>
    where
        E: From<Error>
            + From<Resource>
            + From<fe2o3_protected_service_spawn::RetainedResourceAccessErrorV2>
            + From<fe2o3_protected_service_spawn::native_spawn::ProtectedServiceSpawnErrorV2>,
    {
        self.account
            .with(self.retained, b, |b| {
                Ok(self.trace.with_runtime_backing(b, operation))
            })
            .map_err(E::from)?
    }

    pub(crate) fn runtime_backing_quota() -> Result<Quota> {
        let inner = CompilerTrace::<T>::runtime_backing_quota()?;
        Ok(Quota {
            work: sum(&[LOCAL_WORK, inner.work()])?,
            scratch: sum(&[FRAME, inner.scratch()])?,
        })
    }

    pub(crate) fn runtime_cancellation_quota() -> Result<Quota> {
        let inner = CompilerTrace::<T>::cancellation_quota()?;
        Ok(Quota {
            work: sum(&[LOCAL_WORK, inner.work()])?,
            scratch: sum(&[FRAME, inner.scratch()])?,
        })
    }

    /// Uses the issuer's original cleanup slot without touching compiler/session.
    /// This is not a restart path: the compiler input transfer remains one-use.
    pub(crate) fn cancel_issuer(&mut self) -> Option<CleanupPoll> {
        cancel_issuer_slot(&mut self.issuer)
    }
}

// No root/session/trace or fresh account is available to issuer removal.
fn cancel_issuer_slot<T: Send + 'static>(
    slot: &mut Option<ManagedIssuer<'_, T>>,
) -> Option<CleanupPoll> {
    slot.take().map(|mut issuer| issuer.child.cancel())
}

struct ManagedIssuer<'work, T: Send + 'static> {
    child: Child<T>,
    ready: Ready,
    connection: RootConnection<'work>,
    retained: usize,
}
impl<T: Send + 'static> ManagedIssuer<'_, T> {
    const ENVELOPE: usize = size_of::<(Self, Storage)>()
        - size_of::<Child<T>>()
        - size_of::<Ready>()
        - size_of::<RootConnection<'static>>();
}

fn attempt_storage<T: Send + 'static>(trace: usize, root: usize, issuer: usize) -> Result<usize> {
    sum(&[trace, root, issuer, NativeAttempt::<T>::ENVELOPE])
}

// Readiness alone cannot construct ManagedIssuer. The original held compiler
// trace must pass its scoped identity check and the actual issuer must complete the
// fresh root challenge before this intermediate owner is consumed.
struct ReadyIssuer<'work, T: Send + 'static> {
    child: Child<T>,
    ready: Ready,
    channel: RootChannel<'work>,
    continuity: Quota,
    account: RequestAccount,
}
impl<T: Send + 'static> ReadyIssuer<'_, T> {
    const ENVELOPE: usize = size_of::<(Self, Storage)>()
        - size_of::<Child<T>>()
        - size_of::<Ready>()
        - size_of::<RootChannel<'static>>();
}
impl<T: Send + 'static> fmt::Debug for NativeAttempt<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeAttempt")
            .field("compiler", &self.trace.pid())
            .field("issuer", &self.issuer_pid())
            .field("authority", &"original-trace-and-session-custody-only")
            .finish_non_exhaustive()
    }
}

// Must outlive the OUTERMOST accounting scope: inner input transfer can commit
// successfully before that scope detects a ledger error or unwinds.
struct CompilerCancellation<'a, 'work, T: Send + 'static> {
    trace: &'a mut CompilerTrace<'work, T>,
    committed: bool,
}
impl<T: Send + 'static> Drop for CompilerCancellation<'_, '_, T> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.trace.cancel();
        }
    }
}

impl Prepared {
    /// Consumes actual prepared root authority, the trace and its ONE input
    /// callback. No descriptor intake, PID reopen or AcceptedHandoff is available.
    /// Keep this preparation and the full compiler trace prepaid on the SAME Budget.
    /// The returned charge is GROWTH above BOTH consumed owners, Prepared + trace.
    /// Keep both input reservations and reserve growth before retaining the result.
    /// On error/unwind both foreground inputs are consumed and the original compiler
    /// is cancelled, even when only final outer accounting failed. Never resumes it.
    ///
    /// # Safety
    /// Keep the actual cloning thread alive until compiler and issuer termination
    /// and unresolved cleanup; preserve sole consuming-wait and descriptor/profile custody. The
    /// original T must satisfy its bounded, nonpanicking, independently funded Drop
    /// contract even if issuer cleanup releases the last dependency on another thread.
    /// Cleanup must be the already funded controller carrying this Prepared's root
    /// lifecycle guard; no account may be renewed. Approved runtime enforcement
    /// and privileged publication observation remain separate.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn launch_root_attempt<'work, T: Send + 'static>(
        self,
        mut trace: CompilerTrace<'work, T>,
        timeout: Duration,
        cleanup: &mut Cleanup,
        b: &mut Budget<'work>,
    ) -> Result<(NativeAttempt<'work, T>, Storage)> {
        let trace_storage = trace.retained_storage();
        let mut attempt = CompilerCancellation {
            trace: &mut trace,
            committed: false,
        };
        let input = sum(&[self.retained_storage(), trace_storage])?;
        let (root, issuer, continuity, account, retained, growth) =
            b.with_prepaid_scope(input, 8, LOCAL_WORK, FRAME, |b| -> Result<_> {
                let deadline = launch_io::bounded_deadline(timeout)?;
                self.validate_cleanup_guard(cleanup, b)?;
                let (pending, growth) =
                    attempt
                        .trace
                        .with_issuer_inputs(b, |client, peer, pidfd, dependency, b| {
                            let account = RequestAccount::capture(b);
                            // SAFETY: the caller supplies the creator/custody/Drop contract;
                            // these inputs come only from this original confirmed root trace.
                            unsafe {
                                launch_inputs(
                                    self, client, peer, pidfd, dependency, account, deadline,
                                    cleanup, b,
                                )
                            }
                        })?;
                b.reserve_storage(growth.additional_storage())?;
                let ReadyIssuer {
                    child,
                    ready,
                    channel,
                    continuity,
                    account,
                    ..
                } = pending;
                let (root, connection, retained) =
                    attempt
                        .trace
                        .with_observation(b, |original, b| -> Result<_> {
                            let (root, charge) = RootSession::create(original, b)?;
                            b.reserve_storage(charge.additional_storage())?;
                            let (connection, charge) =
                                child.with_resources(b, |p, b| -> Result<_> {
                                    check_deadline(deadline, "issuer root admission")?;
                                    Ok(root.connect_after_readiness(
                                        original,
                                        &child,
                                        p.prepared.trust.policy().policy(),
                                        p.manifest.manifest(),
                                        p.prepared.credentials,
                                        &ready,
                                        channel,
                                        deadline.saturating_duration_since(Instant::now()),
                                        b,
                                    )?)
                                })?;
                            b.reserve_storage(charge.additional_storage())?;
                            let retained = sum(&[
                                child.retained_storage(),
                                ready.retained_storage(),
                                connection.retained_storage(),
                                ManagedIssuer::<T>::ENVELOPE,
                            ])?;
                            b.reserve_storage(ManagedIssuer::<T>::ENVELOPE)?;
                            Ok((root, connection, retained))
                        })?;
                let issuer = ManagedIssuer {
                    child,
                    ready,
                    connection,
                    retained,
                };
                let retained =
                    attempt_storage::<T>(trace_storage, root.retained_storage(), issuer.retained)?;
                b.reserve_storage(NativeAttempt::<T>::ENVELOPE)?;
                let growth = retained.checked_sub(input).ok_or(Resource::Accounting)?;
                Ok((root, issuer, continuity, account, retained, Storage(growth)))
            })?;
        // No fallible work remains after the outer accounting scope succeeds.
        attempt.committed = true;
        drop(attempt);
        Ok((
            NativeAttempt {
                trace,
                root,
                issuer: Some(issuer),
                publication: None,
                retained,
                continuity,
                account,
                _creator: PhantomData,
            },
            growth,
        ))
    }
}

#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn launch_inputs<'work, T: Send + 'static>(
    prepared: Prepared,
    client: Client,
    peer: BorrowedFd<'_>,
    pidfd: BorrowedFd<'_>,
    dependency: Dependency<T>,
    account: RequestAccount,
    deadline: Instant,
    cleanup: &mut Cleanup,
    b: &mut Budget<'work>,
) -> Result<(ReadyIssuer<'work, T>, Storage)> {
    let original = prepared.retained_storage();
    let floor = sum(&[original, dependency.retained_storage()])?;
    b.with_prepaid_scope(
        floor,
        8,
        LOCAL_WORK,
        FRAME + launch_io::ATTEMPT_SCRATCH + launch_io::PIPE_ATTEMPT_SCRATCH,
        |b| {
            prepared.revalidate(b)?;
            let continuity = prepared.issuer_continuity_quota::<T>()?;
            let credentials = prepared.credentials;
            let (anchor, charge) = prepared.anchor.try_clone_for_supervisor(
                prepared.trust.deployment(),
                prepared.trust.policy(),
                b,
            )?;
            b.reserve_storage(charge.additional_storage())?;
            // Service comes from the actual live anchor transfer, not inert deployment input.
            let (manifest, charge) = Manifest::new(
                client,
                anchor.service(),
                prepared.trust.policy().policy(),
                b,
            )?;
            b.reserve_storage(charge.additional_storage())?;
            let (manifest, charge) = ManifestCap::create(manifest, b)?;
            b.reserve_storage(charge.additional_storage())?;
            let (key, charge) = prepared.trust.key_template().reissue_for_deployed_service(
                prepared.trust.deployment().deployment(),
                prepared.trust.policy().policy(),
                b,
            )?;
            b.reserve_storage(charge.additional_storage())?;
            let retained = payload_storage::<T>(
                original,
                dependency.retained_storage(),
                key.retained_storage(),
                manifest.retained_storage(),
            )?;
            b.reserve_storage(Payload::<T>::ENVELOPE)?;
            let payload = Payload {
                prepared,
                _dependency: dependency,
                key,
                manifest,
                retained,
            };
            b.reserve_storage(Channels::STORAGE)?;
            let channels = Channels::new()?;
            // The compiler already exists and is held at its confirmed exec.
            // Only the issuer end enters the staged descriptor table below.
            let (mut root_channel, charge) = RootChannel::create(b)?;
            b.reserve_storage(charge.additional_storage())?;
            let (stage, stage_charge) = staging::stage(
                &StageInputs {
                    prepared: &payload.prepared,
                    key: &payload.key,
                    manifest: &payload.manifest,
                    retained: payload.retained,
                },
                anchor,
                peer,
                pidfd,
                &channels,
                &root_channel,
                b,
            )?;
            b.reserve_storage(stage_charge)?;
            check_deadline(deadline, "issuer staging")?;
            // SAFETY: final staged image/root/policy/fresh-key/manifest/anchor Files
            // and exact source peer/pidfd duplicates were checked in staging::stage.
            // The COMPLETE payload enters the original funded slot BEFORE clone;
            // it contains no readiness writer. Caller preserves creator and T contracts.
            let (mut child, charge) =
                unsafe { stage.spawn_retaining(credentials, payload, retained, cleanup, b) }?;
            b.reserve_storage(charge.additional_storage())?;
            // Stage contains a readiness writer alias, so it must close before EOF wait.
            drop(stage);
            b.release_storage(stage_charge)?;
            root_channel.close_parent_issuer_endpoint(b)?;
            let Readers {
                ready,
                profile,
                gate,
                exec,
            } = channels.close_child_ends();
            launch_io::await_profile_ready(
                profile.as_fd(),
                exec.as_fd(),
                &mut Observer {
                    child: &child,
                    budget: b,
                },
                deadline,
            )?;
            child.with_resources(b, |p, b| p.prepared.validate_process(child.pid(), b))?;
            launch_io::release_child(
                gate.as_fd(),
                &mut Observer {
                    child: &child,
                    budget: b,
                },
                deadline,
            )?;
            drop(gate);
            launch_io::await_exec_eof(
                exec.as_fd(),
                &mut Observer {
                    child: &child,
                    budget: b,
                },
                deadline,
            )?;
            b.reserve_storage(READY_BYTES)?;
            let bytes = launch_io::receive_ready_pipe::<READY_BYTES, _>(
                ready.as_fd(),
                &mut Observer {
                    child: &child,
                    budget: b,
                },
                deadline,
            )?;
            let (ready_record, charge) = Ready::decode(&bytes, b)?;
            b.reserve_storage(charge.additional_storage())?;
            child.with_resources(b, |p, b| -> Result<()> {
                p.prepared.validate_process(child.pid(), b)?;
                match_ready(&ready_record, child.pid(), &p.manifest, &p.prepared, b)?;
                fe2o3_broker_authority_service::validate_retained_issuer_image_v3(
                    &child,
                    p.prepared.trust.policy().policy(),
                    b,
                )?;
                Ok(())
            })?;
            require_live(&child, b)?;
            check_deadline(deadline, "issuer final validation")?;
            // SAFETY: native exec-status EOF, exact private pipe record PLUS EOF,
            // matches_launch, measured running image, final prepared process/profile
            // and pidfd liveness passed.
            // ALL parent/stage child-end aliases closed before these observations.
            unsafe { child.confirm_exec(b) }?;
            let retained = sum(&[
                child.retained_storage(),
                charge.additional_storage(),
                root_channel.retained_storage(),
                ReadyIssuer::<T>::ENVELOPE,
            ])?;
            let growth = retained.checked_sub(original).ok_or(Resource::Accounting)?;
            b.reserve_storage(ReadyIssuer::<T>::ENVELOPE)?;
            Ok((
                ReadyIssuer {
                    child,
                    ready: ready_record,
                    channel: root_channel,
                    continuity,
                    account,
                },
                Storage(growth),
            ))
        },
    )
}

fn payload_storage<T: Send + 'static>(
    prepared: usize,
    dependency: usize,
    key: usize,
    manifest: usize,
) -> Result<usize> {
    sum(&[prepared, dependency, key, manifest, Payload::<T>::ENVELOPE])
}
fn match_ready(
    ready: &Ready,
    pid: rustix::process::Pid,
    manifest: &ManifestCap,
    prepared: &Prepared,
    b: &mut Budget<'_>,
) -> Result<()> {
    match_ready_records(
        ready,
        launch::pid_u32(pid)?,
        manifest.manifest(),
        prepared.trust.policy().policy(),
        b,
    )
}
fn match_ready_records(
    ready: &Ready,
    pid: u32,
    manifest: &Manifest,
    policy: &fe2o3_compiler_execution_protocol::CompilerExecutionIssuerPolicyV3,
    b: &mut Budget<'_>,
) -> Result<()> {
    if !ready.matches_launch(pid, manifest, policy, b)? {
        return Err(Error::Invalid("issuer readiness names another launch"));
    }
    Ok(())
}
fn require_live<T: Send + 'static>(child: &Child<T>, b: &mut Budget<'_>) -> Result<()> {
    if !child.is_live(b)? {
        return Err(launch_io::Failure::ChildExited("issuer readiness").into());
    }
    Ok(())
}
fn check_deadline(deadline: Instant, phase: &'static str) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(launch_io::Failure::Timeout(phase).into());
    }
    Ok(())
}

mod staging {
    include!("native_root_issuer_staging.rs");
}
#[path = "native_application_currentness.rs"]
pub(crate) mod application_currentness;
mod quota {
    include!("native_root_issuer_quota.rs");
}
#[path = "native_root_issuer_completion.rs"]
mod completion;
#[path = "native_root_enrollment.rs"]
mod enrollment;
pub(crate) use enrollment::OriginalCompilerEnrollment;
#[path = "native_root_issuer_publication_service.rs"]
mod publication_service;
#[cfg(test)]
mod tests {
    include!("native_root_issuer_tests.rs");
}
