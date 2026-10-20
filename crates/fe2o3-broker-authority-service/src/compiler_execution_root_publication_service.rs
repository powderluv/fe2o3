//! Closed authenticated root RPCs over the original runtime and publication slot.
use super::*;
use crate::RootPublicationCustodyV3 as Publication;
use crate::compiler_execution_occurrence::PreparedRootPublicationRetirementV3 as Retirement;
use crate::compiler_execution_root_exchange::{
    PreparedRootControlReplyV3 as Reply, RootControlRequestDispositionV3 as Disposition,
};
use fe2o3_artifact_transaction::{
    INERT_COMPILER_EXECUTION_SUBJECT_BYTES_V3 as SUBJECT_BYTES,
    InertCompilerExecutionSubjectV3 as Subject,
};
use fe2o3_compiler_execution_protocol::{
    CompilerExecutionReceiptCarriageV3 as Carriage, CompilerExecutionRootControlKindV3 as Kind,
};
use fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1 as Enrollment;
use fe2o3_protected_service_spawn::{
    ProtectedServiceCleanupServiceV2 as Cleanup, native_spawn::RootRuntimeTraceV1 as Runtime,
};

const OBSERVED_BYTES: usize = 32 + SUBJECT_BYTES;
const CARRIAGE_WORK: usize = Carriage::COMPOSED_DECODE_WORK;
const CARRIAGE_SCRATCH: usize = Carriage::COMPOSED_DECODE_STORAGE;
const RPC_WORK: usize = ENTRY + 16 * BYTES;
const RPC_FRAME: usize = FRAME + 8 * BYTES + 4 * size_of::<Step<'static>>();

// Tokens retain every borrowed owner on Drop. Commit happens only after the
// complete accounting scope, before transport, and performs no fallible work.
enum Step<'a> {
    Idle,
    Sent,
    Replay,
    Complete(Reply<'a>),
    Observed {
        publication: Publication,
        growth: usize,
        retained: usize,
        reply: Reply<'a>,
    },
    Retired {
        carriage: Carriage,
        growth: usize,
        retained: usize,
        retirement: Retirement<'a>,
        reply: Reply<'a>,
    },
}

// Returning an owned outcome ends every token borrow before mutating the
// session. The original accounting scope must have succeeded before commit.
enum CommittedStep {
    Idle,
    Sent,
    Reply,
    Observed {
        publication: Publication,
        growth: usize,
        retained: usize,
    },
    Retired {
        carriage: Carriage,
        growth: usize,
        retained: usize,
    },
}

impl Step<'_> {
    fn commit(self) -> CommittedStep {
        match self {
            Self::Idle => CommittedStep::Idle,
            Self::Sent => CommittedStep::Sent,
            Self::Replay => CommittedStep::Reply,
            Self::Complete(reply) => {
                reply.commit();
                CommittedStep::Reply
            }
            Self::Observed {
                publication,
                growth,
                retained,
                reply,
            } => {
                reply.commit();
                CommittedStep::Observed {
                    publication,
                    growth,
                    retained,
                }
            }
            Self::Retired {
                carriage,
                growth,
                retained,
                retirement,
                reply,
            } => {
                retirement.commit();
                reply.commit();
                CommittedStep::Retired {
                    carriage,
                    growth,
                    retained,
                }
            }
        }
    }
}

const _: () = assert!(
    4 * size_of::<Step<'static>>() >= size_of::<Step<'static>>() + 2 * size_of::<CommittedStep>()
);

impl<'work> RootControlSessionV3<'work> {
    /// Service at most one authenticated request or one pending reply. Only the
    /// original runtime, measured retained issuer and root-created connection are
    /// accepted. No observation, status, policy provider or retirement receipt
    /// supplied by the compiler can construct this transition.
    /// Enrollment comes from the retained original request; None records joined
    /// original absence. The actual locked capsule must contain a canonical
    /// inventory. Matching coordinates do not establish source/root equivalence.
    ///
    /// A successful Observe installs the sole late publication holder. Retire is
    /// accepted only from the measured issuer after its closed durable join and
    /// exact occurrence comparison. Its full tombstone survives reconnection;
    /// neither replay nor completion reacquires a retired publication.
    ///
    /// Return unreserved growth above the root's previous charge. Keep all input
    /// owners prepaid and use the original account/address. Any error or unwind
    /// poisons publication service: cancel the original attempt and keep its
    /// funded cleanup owner until retirement. A new connection cannot clear it.
    #[allow(clippy::too_many_arguments)]
    pub fn service_publication<T: Send + 'static>(
        &mut self,
        connection: &mut RootConnectionV3<'work>,
        runtime: &mut Runtime<'work>,
        issuer: &Child<T>,
        policy: &Policy,
        manifest: &Manifest,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        enrollment: &Option<Enrollment>,
        b: &mut Budget<'_>,
    ) -> Result<RootConnectionStorageV3> {
        start_publication_call(&mut self.publication_failed, b)?;
        let inputs = issuer.retained_storage().max(sum(&[
            policy.retained_storage(),
            manifest.retained_storage(),
        ])?);
        let floor = sum(&[
            self.retained,
            connection.retained,
            runtime.retained_storage(),
            inputs,
            size_of::<Option<Enrollment>>(),
        ])?;
        let step = b.with_prepaid_scope(floor, 0, RPC_WORK - ENTRY, RPC_FRAME, |b| {
            self.prepare_publication_step(
                connection,
                runtime,
                issuer,
                policy,
                manifest,
                cleanup,
                maximum_handoff_bytes,
                enrollment,
                b,
            )
        })?;
        let growth = match step.commit() {
            CommittedStep::Idle => 0,
            CommittedStep::Sent => {
                connection.reply_pending = false;
                0
            }
            CommittedStep::Reply => {
                connection.reply_pending = true;
                0
            }
            CommittedStep::Observed {
                publication,
                growth,
                retained,
            } => {
                self.publication = Some(publication);
                self.retained = retained;
                connection.reply_pending = true;
                growth
            }
            CommittedStep::Retired {
                carriage,
                growth,
                retained,
            } => {
                self.retirement = Some(carriage);
                self.retained = retained;
                connection.reply_pending = true;
                growth
            }
        };
        self.publication_failed = false;
        Ok(RootConnectionStorageV3(growth))
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_publication_step<'a, T: Send + 'static>(
        &'a self,
        connection: &'a mut RootConnectionV3<'work>,
        runtime: &'a mut Runtime<'work>,
        issuer: &Child<T>,
        policy: &Policy,
        manifest: &Manifest,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        enrollment: &Option<Enrollment>,
        b: &mut Budget<'_>,
    ) -> Result<Step<'a>> {
        self.check_account(b)?;
        connection.channel.validate_root_endpoint(b)?;
        if connection.reply_pending {
            connection
                .validate_established_issuer_authentication(self, issuer, policy, manifest, b)?;
            let request = connection
                .replay
                .request(b)?
                .ok_or(Error::Refused("root reply has no retained request"))?;
            self.validate_completed_request(request, runtime, enrollment, b)?;
            let reply = connection
                .replay
                .reply(b)?
                .ok_or(Error::Refused("root reply is not committed"))?;
            return Ok(
                if connection
                    .channel
                    .send_packet(reply.canonical_bytes(), b)?
                    .is_some()
                {
                    Step::Sent
                } else {
                    Step::Idle
                },
            );
        }
        let sender = launch_io::MessageSender::new(
            connection.issuer.as_raw_pid(),
            connection.credentials.uid(),
            connection.credentials.gid(),
        );
        let Some(bytes) = connection.channel.receive_packet(sender, b)? else {
            return Ok(Step::Idle);
        };
        b.reserve_storage(BYTES)?;
        connection.validate_established_issuer_authentication(self, issuer, policy, manifest, b)?;
        if !runtime.matches_original_identity(&self.original, b)? {
            return Err(Error::Refused("root RPC changed original runtime"));
        }
        let (request, charge) = Record::decode(&bytes, b)?;
        b.reserve_storage(charge.additional_storage())?;
        match connection.replay.accept(request, b)? {
            Disposition::Pending => return Err(Error::Refused("root RPC is already pending")),
            Disposition::Replay => {
                self.validate_completed_request(
                    connection
                        .replay
                        .request(b)?
                        .ok_or(Error::Refused("missing replay request"))?,
                    runtime,
                    enrollment,
                    b,
                )?;
                return Ok(Step::Replay);
            }
            Disposition::Accepted => {}
        }
        let pending = connection
            .replay
            .pending(b)?
            .ok_or(Error::Refused("missing accepted root request"))?;
        // Keep an independently charged inert record while borrowing the mutable
        // replay slot for final preparation; only the retained request sets order.
        let (request, charge) = Record::decode(pending.canonical_bytes(), b)?;
        b.reserve_storage(charge.additional_storage())?;
        match request.kind() {
            Kind::Observe => {
                if !request.payload().is_empty() || self.retirement.is_some() {
                    return Err(Error::Refused(
                        "root observation was already started or retired",
                    ));
                }
                if let Some(publication) = &self.publication {
                    publication.revalidate_runtime_with_enrollment(runtime, enrollment, b)?;
                    let payload = observed_payload(publication);
                    let (reply, charge) = Record::reply(&request, &payload, b)?;
                    b.reserve_storage(charge.additional_storage())?;
                    return Ok(Step::Complete(
                        connection.replay.prepare_complete(reply, b)?,
                    ));
                }
                runtime
                    .with_task_observation(b, |original, b| self.validate_original(original, b))?;
                let (publication, growth) = Publication::observe_runtime_with_limit(
                    runtime,
                    cleanup,
                    maximum_handoff_bytes,
                    b,
                )?;
                b.reserve_storage(growth)?;
                publication.revalidate_runtime_with_enrollment(runtime, enrollment, b)?;
                let retained = sum(&[self.retained, growth])?;
                let payload = observed_payload(&publication);
                let (reply, charge) = Record::reply(&request, &payload, b)?;
                b.reserve_storage(charge.additional_storage())?;
                let reply = connection.replay.prepare_complete(reply, b)?;
                Ok(Step::Observed {
                    publication,
                    growth,
                    retained,
                    reply,
                })
            }
            Kind::Validate => {
                let publication = self.current_publication()?;
                if request.payload() != observed_payload(publication) {
                    return Err(Error::Refused(
                        "root validation changed original occurrence",
                    ));
                }
                publication.revalidate_runtime_with_enrollment(runtime, enrollment, b)?;
                let (reply, charge) = Record::reply(&request, request.payload(), b)?;
                b.reserve_storage(charge.additional_storage())?;
                Ok(Step::Complete(
                    connection.replay.prepare_complete(reply, b)?,
                ))
            }
            Kind::Retire => {
                if let Some(retired) = &self.retirement {
                    require_same_retirement(retired, request.payload())?;
                    let (reply, charge) =
                        Record::reply(&request, retired.identity().as_bytes(), b)?;
                    b.reserve_storage(charge.additional_storage())?;
                    return Ok(Step::Complete(
                        connection.replay.prepare_complete(reply, b)?,
                    ));
                }
                let publication = self.current_publication()?;
                let (carriage, charge) =
                    Carriage::decode_in_original_account_v3(request.payload(), b)?;
                let growth = charge.additional_storage();
                b.reserve_storage(growth)?;
                require_retirement_join(publication, &carriage, policy)?;
                publication.revalidate_runtime_with_enrollment(runtime, enrollment, b)?;
                let retained = sum(&[self.retained, growth])?;
                let (reply, charge) = Record::reply(&request, carriage.identity().as_bytes(), b)?;
                b.reserve_storage(charge.additional_storage())?;
                // SAFETY: the actual measured issuer, authenticated on this
                // original bound connection, sends Retire only AFTER its closed
                // Ledger::retirement_carriage durable Worker/anchor/Ready/ACK join.
                // All canonical policy, Subject and occurrence coordinates were
                // matched above to this actual still-locked publication. No
                // unauthenticated receipt or compiler packet can reach this call.
                #[allow(unsafe_code)]
                let retirement =
                    unsafe { publication.prepare_durable_retirement_runtime(runtime, b) }?.ok_or(
                        Error::Refused("root publication retirement barrier is busy"),
                    )?;
                let reply = connection.replay.prepare_complete(reply, b)?;
                Ok(Step::Retired {
                    carriage,
                    growth,
                    retained,
                    retirement,
                    reply,
                })
            }
            Kind::Reconcile => Err(Error::Refused(
                "root handshake cannot be replayed as a publication RPC",
            )),
        }
    }

    fn current_publication(&self) -> Result<&Publication> {
        if self.retirement.is_some() {
            return Err(Error::Refused("root publication is already retired"));
        }
        self.publication
            .as_ref()
            .ok_or(Error::Refused("root publication was never observed"))
    }

    fn validate_completed_request(
        &self,
        request: &Record,
        runtime: &Runtime<'_>,
        enrollment: &Option<Enrollment>,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        if !runtime.matches_original_identity(&self.original, b)? {
            return Err(Error::Refused("root replay changed original runtime"));
        }
        match request.kind() {
            Kind::Observe | Kind::Validate => self
                .current_publication()?
                .revalidate_runtime_with_enrollment(runtime, enrollment, b)
                .map_err(Into::into),
            Kind::Retire => require_same_retirement(
                self.retirement
                    .as_ref()
                    .ok_or(Error::Refused("root has no durable retirement tombstone"))?,
                request.payload(),
            ),
            Kind::Reconcile => Err(Error::Refused("root publication replay is not a handshake")),
        }
    }

    /// Inert original Subject after durable retirement, including after actual
    /// task exit. This never reconstructs custody or proves successful execution;
    /// completion must separately check the original wait and whole-tree terminal.
    pub fn retired_publication_subject<'a>(
        &'a self,
        runtime: &Runtime<'_>,
        b: &mut Budget<'_>,
    ) -> Result<&'a Subject> {
        let floor = sum(&[self.retained, runtime.retained_storage()])?;
        b.with_prepaid_scope(floor, ENTRY, RPC_WORK, RPC_FRAME, |b| {
            self.check_account(b)?;
            if self.publication_failed || !runtime.matches_original_identity(&self.original, b)? {
                return Err(Error::Refused("root retirement lost its original session"));
            }
            let publication = self
                .publication
                .as_ref()
                .ok_or(Error::Refused("missing original publication"))?;
            let carriage = self
                .retirement
                .as_ref()
                .ok_or(Error::Refused("publication was not durably retired"))?;
            if carriage.publication().compiler_occurrence_identity() != *publication.identity()
                || carriage.request().subject().canonical_bytes()
                    != publication.subject().canonical_bytes()
            {
                return Err(Error::Refused(
                    "retirement tombstone changed original publication",
                ));
            }
            Ok(publication.subject())
        })
    }

    /// Conservative single-poll bound; unused operations are never charged.
    /// Keep every original input and returned growth prepaid separately. This
    /// neither increases the account cap nor renews a deadline/attempt budget.
    pub fn publication_service_quota(
        issuer_image_length: u64,
        maximum_handoff_bytes: usize,
    ) -> Result<RootConnectionQuotaV3> {
        let connection = RootConnectionV3::validation_quota(issuer_image_length)?;
        let observe = Publication::observation_quota(maximum_handoff_bytes)?;
        let validate = Publication::maximum_enrollment_revalidation_quota(maximum_handoff_bytes)?;
        let retire = Publication::maximum_retirement_quota(maximum_handoff_bytes)?;
        Ok(RootConnectionQuotaV3 {
            work: sum(&[
                RPC_WORK,
                Channel::WORK,
                2 * Channel::PACKET_WORK,
                connection.work(),
                3 * Replay::WORK,
                3 * CODEC_WORK,
                2 * Runtime::IDENTITY_COMPARISON_WORK,
                Runtime::ROOT_OBSERVATION_WORK,
                Original::VIEW_WORK,
                Self::VALIDATE_WORK,
                observe.work(),
                2 * validate.work(),
                CARRIAGE_WORK,
                retire.work(),
            ])?,
            scratch: sum(&[
                RPC_FRAME,
                Channel::SCRATCH,
                2 * Channel::PACKET_SCRATCH,
                connection.scratch(),
                3 * Replay::SCRATCH,
                BYTES,
                3 * CODEC_SCRATCH,
                Original::VIEW_SCRATCH,
                Self::VALIDATE_SCRATCH,
                observe.scratch(),
                2 * validate.scratch(),
                CARRIAGE_SCRATCH,
                retire.scratch(),
            ])?,
        })
    }

    pub const fn retired_publication_subject_quota() -> RootConnectionQuotaV3 {
        RootConnectionQuotaV3 {
            work: RPC_WORK + Runtime::IDENTITY_COMPARISON_WORK,
            scratch: RPC_FRAME,
        }
    }
}

fn start_publication_call(failed: &mut bool, b: &mut Budget<'_>) -> Result<()> {
    let previous = std::mem::replace(failed, true);
    b.charge_work(ENTRY)?;
    if previous {
        return Err(Error::Refused("root publication session is poisoned"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_only_steps_commit_without_fabricating_publication() {
        assert!(matches!(Step::Idle.commit(), CommittedStep::Idle));
        assert!(matches!(Step::Sent.commit(), CommittedStep::Sent));
        assert!(matches!(Step::Replay.commit(), CommittedStep::Reply));
    }

    #[test]
    fn publication_entry_poison_survives_debit_refusal_and_unwind() {
        for available in [0, ENTRY - 1, ENTRY, 2 * ENTRY] {
            let mut work = Work::new(available);
            let mut b = Budget::new(&mut work, 0);
            let mut failed = false;
            let result = start_publication_call(&mut failed, &mut b);
            assert_eq!(result.is_ok(), available >= ENTRY);
            assert!(failed);
            assert!(start_publication_call(&mut failed, &mut b).is_err());
            assert!(failed);
            assert_eq!(b.storage(), 0);
            if available == 2 * ENTRY {
                assert_eq!(b.work(), 2 * ENTRY);
            }
        }
        let mut work = Work::new(2 * ENTRY);
        let mut b = Budget::new(&mut work, 0);
        let mut failed = false;
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            start_publication_call(&mut failed, &mut b).unwrap();
            panic!("prepared operation unwound");
        }));
        assert!(unwind.is_err() && failed);
        assert!(start_publication_call(&mut failed, &mut b).is_err());
        assert_eq!(b.work(), 2 * ENTRY);
    }

    #[test]
    fn publication_quote_covers_original_owner_and_single_slot_operations() {
        let limit = 1024 * 1024;
        let q = RootControlSessionV3::publication_service_quota(4096, limit).unwrap();
        let observe = Publication::observation_quota(limit).unwrap();
        let validate = Publication::maximum_revalidation_quota(limit).unwrap();
        let retire = Publication::maximum_retirement_quota(limit).unwrap();
        assert!(q.work() >= RPC_WORK + observe.work() + 2 * validate.work() + retire.work());
        assert!(q.work() >= RPC_WORK + Carriage::COMPOSED_DECODE_WORK);
        assert!(q.scratch() >= RPC_FRAME + Carriage::COMPOSED_DECODE_STORAGE);
        assert!(
            q.scratch()
                >= RPC_FRAME + observe.scratch() + 2 * validate.scratch() + retire.scratch()
        );
        let larger = RootControlSessionV3::publication_service_quota(8192, 2 * limit).unwrap();
        assert!(larger.work() > q.work() && larger.scratch() > q.scratch());
        for (image, bytes) in [(0, limit), (u64::MAX, limit), (4096, 0), (4096, usize::MAX)] {
            assert!(RootControlSessionV3::publication_service_quota(image, bytes).is_err());
        }
        let subject = RootControlSessionV3::retired_publication_subject_quota();
        assert_eq!(subject.work(), RPC_WORK + Runtime::IDENTITY_COMPARISON_WORK);
        assert_eq!(subject.scratch(), RPC_FRAME);
    }
}

fn observed_payload(publication: &Publication) -> [u8; OBSERVED_BYTES] {
    let mut payload = [0; OBSERVED_BYTES];
    payload[..32].copy_from_slice(publication.identity());
    payload[32..].copy_from_slice(publication.subject().canonical_bytes());
    payload
}

fn require_retirement_join(
    publication: &Publication,
    carriage: &Carriage,
    policy: &Policy,
) -> Result<()> {
    if carriage.policy().canonical_bytes() != policy.canonical_bytes()
        || carriage.publication().compiler_occurrence_identity() != *publication.identity()
        || carriage.request().subject().canonical_bytes() != publication.subject().canonical_bytes()
    {
        return Err(Error::Refused(
            "root retirement changed policy or original occurrence",
        ));
    }
    Ok(())
}

// This inert comparison is used only after established measured-peer admission;
// a digest match, unknown root, or lost tombstone cannot authorize retirement.
fn require_same_retirement(retired: &Carriage, payload: &[u8]) -> Result<()> {
    if payload != retired.canonical_bytes() {
        return Err(Error::Refused(
            "root retirement replay changed durable carriage",
        ));
    }
    Ok(())
}
