//! Native content proof controller. Currentness remains an independent host gate.
use super::{
    Activation, ActiveNativeApplication, CAPSULE_BYTES, Capsule, StagedNativeApplication, control,
    evidence,
    resources::{Policy, Tools, read_sealed_file},
    transport,
};
use crate::{NativeApplicationProofCustodianDeploymentV1 as Config, other, require, wire};
use fe2o3_broker_authority_service::LiveClientPidfdIdentityV2 as Client;
use fe2o3_compiler_execution_protocol::{
    CompilerExecutionReceiptCarriageV3 as Carriage,
    NATIVE_PROOF_CUSTODIAN_CONFIGURATION_BYTES_V1 as CONFIG_BYTES,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrOwnedVerificationResourceBudgetV1 as OwnedBudget,
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget, CanonicalKernelIrWorkBudgetV1 as Work,
};
use fe2o3_runtime_protocol::{
    MAX_CONDITIONAL_WORKER_READINESS_BYTES_V5, NATIVE_APPLICATION_PROOF_MAX_PAYLOAD_BYTES_V1,
    NativeApplicationProofEvidenceV1 as Evidence, NativeApplicationProofInputsV1 as Inputs,
    NativeApplicationProofKindV1 as Kind, NativeApplicationProofMessageV1 as Message,
    NativeApplicationProofSessionV1 as Session,
};
use std::{
    fs::File,
    io,
    os::{
        fd::{FromRawFd, OwnedFd},
        unix::fs::MetadataExt,
    },
    time::{Duration, Instant},
};

mod content;
const CONTROLLER_WORK: usize = 1_000_000_000_000;
const CONTROLLER_STORAGE: usize = 2 * 1024 * 1024 * 1024;

fn controller_account() -> OwnedBudget {
    OwnedBudget::new(Work::new(CONTROLLER_WORK), CONTROLLER_STORAGE)
}

/// Sole fixed native secure-entry. The one initial child account lasts until
/// process termination; exhaustion never renews it. This logical account is not
/// the root account or an aggregate RSS guarantee for analyzer/solver processes.
/// The analyzer uses its separately pinned, bounded authenticated execution domain.
///
/// # Safety
/// Invoke once before unrelated descriptor use. Original slots 3..=9 exclusively
/// transfer root-sealed config/capsule, app/Cargo pidfds, app/root peers and the
/// independent root-sealed policy. No prior Rust owner/borrow may exist.
#[doc(hidden)]
pub unsafe fn run_inherited_native_application_proof_controller_v1() -> io::Result<()> {
    let mut account = controller_account();
    account.with_budget(|budget| {
        // SAFETY: the sole entry forwards the unchanged original descriptors.
        unsafe { run_inherited_controller(budget) }
    })
}

unsafe fn run_inherited_controller(mut budget: &mut Budget<'_>) -> io::Result<()> {
    budget.charge_work(64 * 1024).map_err(other)?;
    budget.reserve_storage(64 * 1024).map_err(other)?;
    let mut args = std::env::args_os();
    require(
        args.next().is_some_and(|s| s == "fe2o3-protected-service")
            && args.next().is_none()
            && std::env::vars_os().next().is_none(),
        "native controller entry profile",
    )?;
    for fd in 3..=9 {
        // SAFETY: inspection does not take ownership.
        require(
            unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0,
            "missing native controller slot",
        )?;
    }
    // SAFETY: this sole secure entry takes each transferred slot exactly once.
    let [config, capsule, app, cargo, peer, root, policy] =
        unsafe { [3, 4, 5, 6, 7, 8, 9].map(|fd| OwnedFd::from_raw_fd(fd)) };
    for fd in [&config, &capsule, &app, &cargo, &peer, &root, &policy] {
        rustix::io::fcntl_setfd(fd, rustix::io::FdFlags::CLOEXEC)?;
    }
    let (config_file, capsule_file, policy_file) =
        (File::from(config), File::from(capsule), File::from(policy));
    let ids = [&config_file, &capsule_file, &policy_file]
        .map(|f| f.metadata().map(|m| (m.dev(), m.ino())));
    let [a, b, c] = ids;
    let (a, b, c) = (a?, b?, c?);
    require(a != b && a != c && b != c, "native root inputs alias")?;
    budget
        .reserve_storage(3 * size_of::<File>() + 2 * Client::FD_STORAGE + 2 * size_of::<OwnedFd>())
        .map_err(other)?;
    let (config_bytes, s) = read_sealed_file(&config_file, (0, 0), CONFIG_BYTES, &mut budget)?;
    budget.reserve_storage(s).map_err(other)?;
    let (config, s) = Config::decode(&config_bytes, &mut budget)?;
    budget
        .reserve_storage(s.additional_storage())
        .map_err(other)?;
    let (capsule_bytes, s) = read_sealed_file(&capsule_file, (0, 0), CAPSULE_BYTES, &mut budget)?;
    budget.reserve_storage(s).map_err(other)?;
    let (capsule, s) = Capsule::decode(&capsule_bytes, &mut budget)?;
    budget.reserve_storage(s).map_err(other)?;
    let parent = (capsule.parent as i32, 0, 0);
    require(
        rustix::process::getppid().map(|p| p.as_raw_pid()) == Some(parent.0),
        "native controller parent changed",
    )?;
    let mut signal = 0;
    // SAFETY: read-only prctl writes one initialized scalar.
    require(
        unsafe { libc::prctl(libc::PR_GET_PDEATHSIG, &raw mut signal, 0, 0, 0) } == 0
            && signal == libc::SIGKILL,
        "native controller parent-death signal missing",
    )?;
    let root = wire::ControlEndpoint::admit(root)?;
    let (policy, s) = Policy::admit(policy_file, &config, &mut budget)?;
    budget.reserve_storage(s).map_err(other)?;
    let (tools, s) = Tools::open(&config, &mut budget)?;
    budget.reserve_storage(s).map_err(other)?;
    let credentials = config.credentials()?;
    let (session, s) = Session::new(
        capsule.transcript,
        config.identity(),
        capsule.nonce,
        (std::process::id(), credentials.uid(), credentials.gid()),
        &mut budget,
    )
    .map_err(other)?;
    budget
        .reserve_storage(s.additional_storage())
        .map_err(other)?;
    let (mut staged, s) =
        StagedNativeApplication::admit(capsule, app, cargo, peer, &config, &mut budget)?;
    budget.reserve_storage(s).map_err(other)?;
    let deadline = Instant::now() + Duration::from_secs(300);
    require(
        control::try_send(
            &root,
            control::Kind::Ready,
            session.nonce(),
            session.identity(),
            &mut budget,
        )?,
        "native root Ready pending",
    )?;
    let application = loop {
        require(Instant::now() < deadline, "native activation deadline")?;
        match staged.try_activate(&root, &session, &mut budget)? {
            Activation::Active(active) => break active,
            Activation::Pending(pending) => {
                staged = pending;
                std::thread::park_timeout(Duration::from_millis(10));
            }
        }
    };
    // From activation onwards failures retain original process/input/resource custody.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> io::Result<()> {
        tools.revalidate(&mut budget)?;
        policy.revalidate(&mut budget)?;
        require(
            control::try_send(
                &root,
                control::Kind::Activated,
                session.nonce(),
                session.identity(),
                &mut budget,
            )?,
            "native root Activated pending",
        )?;
        let (active, s) = Message::active(&session, &mut budget).map_err(other)?;
        budget
            .reserve_storage(s.retained_storage())
            .map_err(other)?;
        require(
            transport::try_send(application.peer(&mut budget)?, &active, &mut budget)?,
            "native Active pending",
        )?;
        drop(active);
        budget
            .release_storage(s.retained_storage())
            .map_err(other)?;
        let (inputs, readiness, payload) =
            receive_inputs(&application, &session, deadline, &mut budget)?;
        // Content/proof owners are retained in nested stack scopes, never rebuilt from evidence.
        content::execute(
            &application,
            &session,
            &root,
            parent,
            &policy,
            &tools,
            &config,
            &inputs,
            &readiness,
            &payload,
            deadline,
            &mut budget,
        )
    }));
    let _ = std::hint::black_box(result);
    quarantine((
        &application,
        &policy,
        &tools,
        &config_file,
        &capsule_file,
        &root,
        &session,
        &budget,
    ))
}

fn sender(application: &ActiveNativeApplication<'_>) -> (i32, u32, u32) {
    let client = application
        .capsule()
        .binding
        .compiler_handoff()
        .launch_manifest()
        .client();
    (client.pid() as i32, client.uid(), client.gid())
}
fn receive_inputs<'work>(
    application: &ActiveNativeApplication<'work>,
    session: &Session,
    deadline: Instant,
    budget: &mut Budget<'work>,
) -> io::Result<(Inputs, File, File)> {
    loop {
        require(Instant::now() < deadline, "native proof request deadline")?;
        let peer = application.peer(budget)?;
        if let Some((mut received, s)) =
            transport::try_receive(peer, sender(application), session, budget)?
        {
            budget.reserve_storage(s).map_err(other)?;
            require(
                received.message.kind() == Kind::Request && received.message.sequence() == 1,
                "native controller expected one Request",
            )?;
            let (inputs, charge) = received.message.decode_inputs(budget).map_err(other)?;
            budget
                .reserve_storage(charge.retained_storage())
                .map_err(other)?;
            inputs
                .check_registration(&application.capsule().binding, budget)
                .map_err(other)?;
            let first = File::from(
                received.rights[0]
                    .take()
                    .ok_or_else(|| io::Error::other("missing native readiness right"))?,
            );
            let second = File::from(
                received.rights[1]
                    .take()
                    .ok_or_else(|| io::Error::other("missing native payload right"))?,
            );
            let (a, b) = (first.metadata()?, second.metadata()?);
            require(
                (a.dev(), a.ino()) != (b.dev(), b.ino()),
                "native proof inputs alias",
            )?;
            budget
                .reserve_storage(2 * size_of::<File>())
                .map_err(other)?;
            drop(received);
            budget.release_storage(s).map_err(other)?;
            return Ok((inputs, first, second));
        }
        std::thread::park_timeout(Duration::from_millis(10));
    }
}

fn quarantine(custody: impl Sized) -> ! {
    loop {
        std::hint::black_box(&custody);
        std::thread::park_timeout(Duration::from_secs(1));
    }
}

#[derive(Default)]
struct ProbeSequence {
    last: u64,
}
impl ProbeSequence {
    fn accept(&mut self, message: &Message) -> io::Result<()> {
        require(
            message.kind() == Kind::Probe
                && message.sequence() > self.last
                && message.sequence() >= 2
                && message.sequence() != u64::MAX,
            "native retained probe transition",
        )?;
        self.last = message.sequence();
        Ok(())
    }
}

fn serve<'work>(
    application: &ActiveNativeApplication<'work>,
    session: &Session,
    root: &wire::ControlEndpoint,
    parent: (i32, u32, u32),
    tools: &Tools,
    evidence: &Evidence,
    mut revalidate_content: impl FnMut(&mut Budget<'work>) -> io::Result<()>,
    budget: &mut Budget<'work>,
) -> io::Result<()> {
    let mut sequence = ProbeSequence::default();
    loop {
        if let Some(kind) =
            control::try_receive(root, parent, session.nonce(), session.identity(), budget)?
        {
            require(
                kind == control::Kind::Probe,
                "native root retained transition",
            )?;
            tools.revalidate(budget)?;
            revalidate_content(budget)?;
            require(
                control::try_send(
                    root,
                    control::Kind::Retained,
                    session.nonce(),
                    session.identity(),
                    budget,
                )?,
                "native root Retained pending",
            )?;
        }
        if let Some((received, charge)) = transport::try_receive(
            application.peer(budget)?,
            sender(application),
            session,
            budget,
        )? {
            budget.reserve_storage(charge).map_err(other)?;
            sequence.accept(&received.message)?;
            tools.revalidate(budget)?;
            revalidate_content(budget)?;
            let (answer, s) =
                Message::retained(session, sequence.last, evidence, budget).map_err(other)?;
            budget
                .reserve_storage(s.retained_storage())
                .map_err(other)?;
            require(
                transport::try_send(application.peer(budget)?, &answer, budget)?,
                "native Retained pending",
            )?;
            drop(answer);
            drop(received);
            budget
                .release_storage(charge + s.retained_storage())
                .map_err(other)?;
        }
        std::thread::park_timeout(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_runtime_protocol::NativeApplicationSessionTranscriptV1 as Transcript;

    #[test]
    fn native_controller_has_one_owned_account_for_composed_codecs() {
        let mut account = controller_account();
        let floor = 37 + Budget::STORAGE_WINDOW_SCRATCH_V1;
        account.with_budget(|budget| {
            budget.charge_work(17).unwrap();
            budget.reserve_storage(floor).unwrap();
            let identity = budget.storage_account_identity_v1().unwrap();
            let ledger = budget.work_ledger_identity_v1();
            budget
                .with_additional_storage_window_v1(23, |budget| {
                    assert_eq!(budget.storage_account_identity_v1(), Some(identity));
                    assert!(budget.work_ledger_identity_v1() == ledger);
                    budget.reserve_storage(23)?;
                    Ok::<_, fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceErrorV1>(())
                })
                .unwrap();
            assert_eq!(budget.storage_limit(), CONTROLLER_STORAGE);
            assert_eq!(budget.storage(), floor + 23);
            assert_eq!(budget.work(), 17 + Budget::STORAGE_WINDOW_WORK_V1);
            assert_eq!(budget.storage_account_identity_v1(), Some(identity));
            assert!(budget.work_ledger_identity_v1() == ledger);
        });
        account.with_budget(|budget| {
            assert_eq!(budget.storage(), floor + 23);
            assert_eq!(budget.work(), 17 + Budget::STORAGE_WINDOW_WORK_V1);
        });
    }

    #[test]
    fn native_controller_account_keeps_window_denials_across_borrows() {
        let mut account = controller_account();
        let floor = 37 + Budget::STORAGE_WINDOW_SCRATCH_V1;
        account.with_budget(|budget| {
            budget.reserve_storage(floor).unwrap();
            assert!(
                budget
                    .with_additional_storage_window_v1(23, |budget| budget.reserve_storage(24))
                    .is_err()
            );
            assert_eq!(budget.storage(), floor);
            assert_eq!(budget.failed_storage(), Some(floor + 24));
            assert!(budget.charge_work(CONTROLLER_WORK).is_err());
        });
        account.with_budget(|budget| {
            assert_eq!(budget.storage(), floor);
            assert_eq!(budget.failed_storage(), Some(floor + 24));
            assert_eq!(budget.work(), Budget::STORAGE_WINDOW_WORK_V1);
            assert_eq!(
                budget.failed_work(),
                Some(CONTROLLER_WORK + Budget::STORAGE_WINDOW_WORK_V1),
            );
            assert_eq!(budget.storage_limit(), CONTROLLER_STORAGE);
        });
    }

    #[test]
    fn native_retained_phase_refuses_repeat_and_non_probe_without_advancing() {
        let mut work = Work::new(1_000_000);
        let mut budget = Budget::new(&mut work, 1_000_000);
        let (transcript, charge) =
            Transcript::from_untrusted_parts([1; 32], [2; 32], [3; 32], &mut budget).unwrap();
        budget.reserve_storage(charge.additional_storage()).unwrap();
        let (session, charge) = Session::new(
            transcript,
            [4; 32],
            [5; 32],
            (1200, 1001, 1001),
            &mut budget,
        )
        .unwrap();
        budget.reserve_storage(charge.additional_storage()).unwrap();
        let mut sequence = ProbeSequence::default();
        let (active, charge) = Message::active(&session, &mut budget).unwrap();
        budget.reserve_storage(charge.retained_storage()).unwrap();
        assert!(sequence.accept(&active).is_err());
        assert_eq!(sequence.last, 0);
        let (probe, charge) = Message::probe(&session, 2, &mut budget).unwrap();
        budget.reserve_storage(charge.retained_storage()).unwrap();
        sequence.accept(&probe).unwrap();
        assert_eq!(sequence.last, 2);
        assert!(sequence.accept(&probe).is_err());
        assert!(sequence.accept(&active).is_err());
        assert_eq!(sequence.last, 2);
        let (next, charge) = Message::probe(&session, 3, &mut budget).unwrap();
        budget.reserve_storage(charge.retained_storage()).unwrap();
        sequence.accept(&next).unwrap();
        assert_eq!(sequence.last, 3);
        assert!(sequence.accept(&probe).is_err());
    }
}
