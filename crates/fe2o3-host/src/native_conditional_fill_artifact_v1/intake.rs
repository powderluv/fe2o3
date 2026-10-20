//! Original four-descriptor intake, retained publication and native source custody.
use super::abi::{CheckedNativeConditionalFillAbiV1, check_native_conditional_fill_abi_v1};
use fe2o3_artifact_transaction::{
    DurablePublishedHsacoClaimV3 as Claim, NativeCurrentPublicationLimitsV1 as Limits,
    NativeCurrentPublicationV1 as Publication, ProducerIdentity,
    RetainedDurableDirectoryV1 as Directory, WorkerV3PublicationIntentRecordV1 as Record,
};
use fe2o3_compiler_closure_capability::{
    ProductionCompilerExecutionDeploymentV3 as Compiler,
    ProductionNativeApplicationProofProfileV1 as Profile,
};
use fe2o3_compiler_execution_client::{
    RegisteredNativeApplicationCustodianV1 as Registration,
    RetainedApplicationProofEndpointV1 as Endpoint,
};
use fe2o3_compiler_execution_protocol::CompilerExecutionReceiptCarriageV3 as Carriage;
use fe2o3_hsaco_finalize::{
    ConditionalWorkerCompactFinalizerReplayV5 as Transcript,
    PreparedFinalizedConditionalWorkerHsacoV5 as Finalized,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrStorageAccountIdentityV1 as Account,
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
    CanonicalKernelIrWorkLedgerIdentityV1 as Ledger,
};
use fe2o3_runtime_protocol::{
    InertConditionalWorkerReadinessWireV5 as Wire, NativeApplicationRegistrationInputsV1 as Inputs,
    WorkerV3ApplicationHandoffChallengeV1 as Challenge,
    WorkerV3ApplicationInputOccurrenceV1 as InputOccurrence,
    WorkerV3ApplicationOccurrenceV1 as Occurrence,
    WorkerV3ApplicationRegistrationDescriptorsV1 as Descriptors,
};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    fmt,
    mem::size_of,
    os::fd::{AsRawFd, OwnedFd},
    time::Instant,
};
mod content;
mod inherited;
mod io;
mod lifecycle;
pub use lifecycle::{
    CurrentnessReadyNativeConditionalFillApplicationV1, NativeConditionalFillInvocationScopeV1,
    ProvedNativeConditionalFillApplicationV1,
};

#[derive(Debug)]
pub enum NativeConditionalFillIntakeErrorV1 {
    Resource(Resource),
    Rejected(String),
}
type Error = NativeConditionalFillIntakeErrorV1;
type Result<T> = std::result::Result<T, Error>;
impl From<Resource> for Error {
    fn from(value: Resource) -> Self {
        Self::Resource(value)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "native application intake: {self:?}")
    }
}
impl std::error::Error for Error {}
fn failure(error: impl fmt::Display) -> Error {
    Error::Rejected(error.to_string())
}
fn require(condition: bool, reason: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(failure(reason))
    }
}

/// Additional logical charge returned unreserved. Intake includes its consumed
/// descriptors and occurrence; registration preserves all prepaid input floors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConditionalFillIntakeStorageV1(usize);
impl NativeConditionalFillIntakeStorageV1 {
    pub const fn additional_storage(self) -> usize {
        self.0
    }
}

struct Files<'work> {
    publication: Publication<'work>,
    finalized: Finalized,
    transcript: Transcript,
    carriage: Carriage,
    _directory: Directory,
    envelope: OwnedFd,
    // Taken exactly once by the startup transition; root requires ACK pipe EOF.
    acknowledgment: Option<OwnedFd>,
    envelope_stat: rustix::fs::Stat,
    acknowledgment_stat: rustix::fs::Stat,
    ledger: Ledger,
    account: Option<Account>,
    retained: usize,
    failed: Cell<bool>,
}

/// Original descriptors and real content owners before native root registration.
///
/// Neither supplied occurrence bytes nor the content reconstruction authenticates
/// the root peer. The original ACK remains retained and unwritten: only the
/// separately integrated native root protocol may authorize that transition.
/// No legacy envelope decoding, ordinary-family authority or launch conversion.
///
/// ```compile_fail
/// use fe2o3_host::PreparedNativeConditionalFillApplicationV1 as Native;
/// fn duplicate(value: Native<'_>) { let _ = value.clone(); }
/// ```
pub struct PreparedNativeConditionalFillApplicationV1<'work> {
    files: Files<'work>,
    inputs: Inputs,
    endpoint: Endpoint,
    retained: usize,
}

/// Actual native controller registration joined to original application files.
/// Still no startup ACK, current-record query, machine-proof completion or launch.
pub struct RegisteredNativeConditionalFillApplicationV1<'work> {
    files: Files<'work>,
    registration: Registration<'work>,
    retained: usize,
}

impl<'work> PreparedNativeConditionalFillApplicationV1<'work> {
    /// Consume actual owned envelope/directory/ACK/proof descriptors in that order.
    /// Caller must invoke before concurrent descriptor mutation or descendant
    /// creation. The borrowed actual profiles and Producer backing stay prepaid;
    /// every native I/O attempt is finite and all failed charges remain terminal.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_descriptors(
        descriptors: [OwnedFd; 4],
        occurrence: Occurrence,
        challenge: Challenge,
        producer: &ProducerIdentity,
        compiler: &Compiler<'work>,
        profile: &Profile<'work>,
        budget: &mut Budget<'work>,
    ) -> Result<(Self, NativeConditionalFillIntakeStorageV1)> {
        budget.charge_work(4096)?;
        let floor = budget.storage();
        let mut producer_storage = size_of::<ProducerIdentity>();
        producer.visit_retained_heap_storage_v1(|n, w| -> Result<()> {
            budget.charge_work(1)?;
            producer_storage = producer_storage
                .checked_add(n.checked_mul(w).ok_or(Resource::Arithmetic)?)
                .ok_or(Resource::Arithmetic)?;
            Ok(())
        })?;
        let required = compiler
            .retained_storage()
            .map_err(failure)?
            .checked_add(profile.retained_storage().map_err(failure)?)
            .and_then(|n| n.checked_add(producer_storage))
            .ok_or(Resource::Arithmetic)?;
        if floor < required {
            return Err(Resource::Accounting.into());
        }
        let occurrence_storage = size_of::<Occurrence>() + 4 * size_of::<InputOccurrence>();
        budget.reserve_storage(128 * 1024 + occurrence_storage + size_of::<Self>())?;
        profile.revalidate(compiler, budget).map_err(failure)?;
        let numbers = descriptors.each_ref().map(AsRawFd::as_raw_fd);
        let wire_descriptors =
            Descriptors::new(numbers[0], numbers[1], numbers[2], numbers[3]).map_err(failure)?;
        let mut observed = Vec::with_capacity(4);
        for (index, fd) in descriptors.iter().enumerate() {
            observed.push(io::occurrence(fd, (index + 1) as u16)?);
        }
        require(
            observed.iter().enumerate().all(|(i, row)| {
                observed[..i]
                    .iter()
                    .all(|other| other.identity() != row.identity())
            }),
            "native descriptor object alias",
        )?;
        require(
            occurrence.inputs() == observed && occurrence.application() == io::application(budget)?,
            "original application/four-FD occurrence",
        )?;
        let [envelope, directory, acknowledgment, endpoint] = descriptors;
        let envelope_stat = io::regular(
            &envelope,
            fe2o3_runtime_protocol::MAX_CONDITIONAL_WORKER_READINESS_BYTES_V5,
        )?;
        let acknowledgment_stat = io::acknowledgment(&acknowledgment)?;
        let (bytes, read_storage) = io::read(&envelope, envelope_stat.st_size as usize, budget)?;
        require(
            io::same(
                &envelope_stat,
                &rustix::fs::fstat(&envelope).map_err(failure)?,
            ),
            "original readiness changed",
        )?;
        let wire = Wire::decode(&bytes, bytes.len()).map_err(failure)?;
        let record = Record::decode_canonical(wire.record_bytes()).map_err(failure)?;
        let claim = Claim::decode_canonical(wire.claim_bytes()).map_err(failure)?;
        require(
            record.plan() == claim.plan(),
            "native readiness record/claim plan",
        )?;
        let authenticated = content::authenticate(&wire, &record, compiler, profile, budget)?;
        let limits = Limits::new(bytes.len(), record.output_length()).map_err(failure)?;
        io::directory(&directory)?;
        let directory = Directory::admit_service_owned(directory).map_err(failure)?;
        budget.reserve_storage(Publication::DIRECTORY_STORAGE)?;
        let (publication, publication_charge) =
            Publication::recover(&directory, record.attempt(), limits, budget).map_err(failure)?;
        budget.reserve_storage(publication_charge.additional_storage())?;
        require(
            publication.readiness().exact_envelope_bytes() == bytes
                && publication.readiness().published_claim() == &claim,
            "exact canonical native readiness custody",
        )?;
        let (finalized, transcript, carriage) = content::recover(
            authenticated,
            &wire,
            &record,
            &claim,
            &publication,
            producer,
            profile,
            budget,
        )?;
        let (abi, abi_storage) =
            check_native_conditional_fill_abi_v1(finalized.source().recovered_handoff(), budget)
                .map_err(failure)?;
        budget.reserve_storage(abi_storage)?;
        drop(abi);
        budget.release_storage(abi_storage)?;
        drop(bytes);
        budget.release_storage(read_storage)?;
        let endpoint = Endpoint::admit_inherited(endpoint).map_err(failure)?;
        let (inputs, input_charge) = Inputs::new(
            publication.readiness().exact_envelope_bytes(),
            occurrence,
            wire_descriptors,
            challenge,
            budget,
        )
        .map_err(failure)?;
        budget.reserve_storage(input_charge.additional_storage())?;
        publication.revalidate(budget).map_err(failure)?;
        profile.revalidate(compiler, budget).map_err(failure)?;
        let retained = budget
            .storage()
            .checked_sub(floor)
            .ok_or(Resource::Accounting)?;
        // The conservative header/scratch allowance remains part of the logical
        // owner quote, so no transferred component is accidentally undercounted.
        let files = Files {
            publication,
            finalized,
            transcript,
            carriage,
            _directory: directory,
            envelope,
            acknowledgment: Some(acknowledgment),
            envelope_stat,
            acknowledgment_stat,
            ledger: budget.work_ledger_identity_v1(),
            account: budget.storage_account_identity_v1(),
            retained,
            failed: Cell::new(false),
        };
        let value = Self {
            files,
            inputs,
            endpoint,
            retained,
        };
        budget.release_storage(retained)?;
        Ok((value, NativeConditionalFillIntakeStorageV1(retained)))
    }

    pub fn checked_abi(
        &self,
        budget: &mut Budget<'work>,
    ) -> Result<(CheckedNativeConditionalFillAbiV1<'_>, usize)> {
        self.files.revalidate(budget)?;
        check_native_conditional_fill_abi_v1(
            self.files.finalized.source().recovered_handoff(),
            budget,
        )
        .map_err(failure)
    }
    pub const fn retained_storage(&self) -> usize {
        self.retained
    }
    pub const fn grants_launch_authority(&self) -> bool {
        false
    }

    /// Original compiler/profile owners move once into the actual native session.
    /// The ACK is not emitted and ordinary Ready is never accepted as CustodianReady.
    pub fn register_custodian(
        self,
        compiler: Compiler<'work>,
        profile: Profile<'work>,
        deadline: Instant,
        budget: &mut Budget<'work>,
    ) -> Result<(
        RegisteredNativeConditionalFillApplicationV1<'work>,
        NativeConditionalFillIntakeStorageV1,
    )> {
        self.files.revalidate(budget)?;
        let profiles = compiler
            .retained_storage()
            .map_err(failure)?
            .checked_add(profile.retained_storage().map_err(failure)?)
            .ok_or(Resource::Arithmetic)?;
        let (registration, charge) = self
            .endpoint
            .register_native_custodian_pre_ack(self.inputs, compiler, profile, deadline, budget)
            .map_err(failure)?;
        let retained = self
            .retained
            .checked_add(profiles)
            .and_then(|n| n.checked_add(charge.additional_storage()))
            .ok_or(Resource::Arithmetic)?;
        Ok((
            RegisteredNativeConditionalFillApplicationV1 {
                files: self.files,
                registration,
                retained,
            },
            NativeConditionalFillIntakeStorageV1(charge.additional_storage()),
        ))
    }
}

impl<'work> Files<'work> {
    fn revalidate(&self, budget: &mut Budget<'work>) -> Result<()> {
        let result = (|| {
            require(
                !self.failed.get()
                    && self.ledger == budget.work_ledger_identity_v1()
                    && self.account == budget.storage_account_identity_v1()
                    && budget.storage() >= self.retained,
                "original native application account",
            )?;
            budget.charge_work(4096)?;
            self.publication.revalidate(budget).map_err(failure)?;
            require(
                io::same(
                    &io::regular(
                        &self.envelope,
                        self.publication.readiness().exact_envelope_bytes().len(),
                    )?,
                    &self.envelope_stat,
                ),
                "native original descriptor mutation",
            )?;
            if let Some(acknowledgment) = &self.acknowledgment {
                require(
                    io::same(
                        &io::acknowledgment(acknowledgment)?,
                        &self.acknowledgment_stat,
                    ),
                    "native original ACK descriptor mutation",
                )?;
            }
            let (bytes, storage) =
                io::read(&self.envelope, self.envelope_stat.st_size as usize, budget)?;
            require(
                bytes == self.publication.readiness().exact_envelope_bytes()
                    && io::same(
                        &self.envelope_stat,
                        &rustix::fs::fstat(&self.envelope).map_err(failure)?,
                    ),
                "native original readiness bytes changed",
            )?;
            drop(bytes);
            budget.release_storage(storage)?;
            Ok(())
        })();
        if result.is_err() {
            self.failed.set(true);
        }
        result
    }
}

impl<'work> RegisteredNativeConditionalFillApplicationV1<'work> {
    pub fn revalidate(&self, budget: &mut Budget<'work>) -> Result<()> {
        if budget.storage() < self.retained {
            return Err(Resource::Accounting.into());
        }
        self.files.revalidate(budget)?;
        self.registration.revalidate(budget).map_err(failure)
    }
    pub fn registration(&self) -> &Registration<'work> {
        &self.registration
    }
    pub fn finalized(&self) -> &Finalized {
        &self.files.finalized
    }
    pub fn transcript(&self) -> &Transcript {
        &self.files.transcript
    }
    pub fn carriage(&self) -> &Carriage {
        &self.files.carriage
    }
    pub const fn retained_storage(&self) -> usize {
        self.retained
    }
    pub const fn grants_launch_authority(&self) -> bool {
        false
    }
}
