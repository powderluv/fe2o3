//! Real fixed-slot refusals are isolated from the parent test process's FDs.
use super::super::tests::{assert_closed, install, isolated, leave_service_hole};
use super::*;
use fe2o3_compiler_execution_client::COMPILER_EXECUTION_SERVICE_CHILD_FD_V1 as SERVICE_FD;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use std::fs::File;

#[test]
fn native_owned_startup_refusals_close_copies_and_never_reset_the_account() {
    use super::super::{CompilerExecutionStartupInputV1 as Startup, OwnedExecutionInputs};
    use rustix::pipe::{PipeFlags, pipe_with};
    use std::sync::Mutex;

    for (prepaid, limit) in [
        (Admitted::INPUT_STORAGE - 1, usize::MAX),
        (Admitted::INPUT_STORAGE, 0),
        (Admitted::INPUT_STORAGE, usize::MAX),
    ] {
        let (policy_reader, policy) = pipe_with(PipeFlags::CLOEXEC).unwrap();
        let (service_reader, service) = pipe_with(PipeFlags::CLOEXEC).unwrap();
        let startup = Startup(Mutex::new(Some(Ok(OwnedExecutionInputs {
            policy,
            service,
        }))));
        let mut work = Work::new(limit);
        let mut b = Budget::new(&mut work, 4 * 1024 * 1024);
        b.reserve_storage(prepaid).unwrap();
        let ledger = b.work_ledger_identity_v1();
        let result = startup.admit_native(&mut b);
        if prepaid < Admitted::INPUT_STORAGE {
            assert!(matches!(result, Err(Error::Resource(Resource::Accounting))));
        } else if limit == 0 {
            assert!(matches!(result, Err(Error::Resource(Resource::Work(_)))));
        } else {
            assert!(matches!(result, Err(Error::Policy(_))));
        }
        drop(result);
        assert_eq!(b.storage(), prepaid);
        assert!(b.work_ledger_identity_v1() == ledger);
        for reader in [policy_reader, service_reader] {
            assert_eq!(rustix::io::read(reader, &mut [0]).unwrap(), 0);
        }
        assert!(matches!(
            startup.admit_native(&mut b),
            Err(Error::Startup(
                super::super::ProtectedCompilerExecutionErrorV1::InputAlreadyConsumed
            ))
        ));
        assert!(matches!(
            startup.admit(),
            Err(super::super::ProtectedCompilerExecutionErrorV1::InputAlreadyConsumed)
        ));
    }
}

#[test]
fn native_loader_copies_enter_the_same_account_without_consuming_caller_slots() {
    if isolated(concat!(
        module_path!(),
        "::native_loader_copies_enter_the_same_account_without_consuming_caller_slots"
    )) {
        return;
    }
    use super::super::CompilerExecutionStartupInputV1 as Startup;
    use rustix::net::{AddressFamily, SocketFlags, SocketType, socketpair};
    use std::os::fd::{FromRawFd, OwnedFd};

    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, 4 * 1024 * 1024);
    b.reserve_storage(91).unwrap();
    b.charge_work(19).unwrap();
    let policy = valid_native_policy(&mut b);
    let (file, charge) = policy.try_clone_for_transfer(&mut b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (client, _server) = socketpair(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    let client = File::from(client);
    install(&file, POLICY_FD, false);
    install(&client, SERVICE_FD, false);
    // SAFETY: install relinquished these fresh descriptors in this isolated
    // process. The loader borrows them and owns only its private duplicates.
    let originals = unsafe {
        [
            OwnedFd::from_raw_fd(POLICY_FD),
            OwnedFd::from_raw_fd(SERVICE_FD),
        ]
    };
    let startup = Startup::capture();
    b.reserve_storage(Admitted::INPUT_STORAGE).unwrap();
    let floor = b.storage();
    let before_work = b.work();
    let ledger = b.work_ledger_identity_v1();
    let address = &b as *const Budget<'_> as usize;
    let retained = {
        let admitted = startup.admit_native(&mut b).unwrap();
        assert_eq!(admitted.policy.policy(), policy.policy());
        let Admitted {
            policy: retained,
            client,
            enrollment,
        } = admitted;
        assert!(enrollment.is_none());
        let (client, ()) = client
            .prepare::<_, Error>(|budget| {
                assert!(budget.work_ledger_identity_v1() == ledger);
                assert_eq!(budget as *const Budget<'_> as usize, address);
                assert!(budget.storage() >= floor && budget.work() > before_work);
                retained.revalidate(budget)?;
                budget.charge_work(7)?;
                Ok(())
            })
            .unwrap();
        drop(client);
        drop(enrollment);
        retained
    };
    assert_eq!(
        b.storage(),
        floor - Client::PEER_STORAGE + retained.retained_storage() - Policy::FILE_STORAGE
    );
    assert!(b.work_ledger_identity_v1() == ledger && b.work() > before_work + 7);
    drop(retained);
    assert!(matches!(
        startup.admit_native(&mut b),
        Err(Error::Startup(
            super::super::ProtectedCompilerExecutionErrorV1::InputAlreadyConsumed
        ))
    ));
    assert!(matches!(
        startup.admit(),
        Err(super::super::ProtectedCompilerExecutionErrorV1::InputAlreadyConsumed)
    ));
    drop(startup);
    for fd in &originals {
        rustix::fs::fstat(fd).unwrap();
        assert!(
            rustix::io::fcntl_getfd(fd)
                .unwrap()
                .contains(rustix::io::FdFlags::CLOEXEC)
        );
    }
    drop(originals);
    assert_closed();
    policy.revalidate(&mut b).unwrap();
}

#[test]
fn selected_native_session_keeps_the_original_budget_and_consumes_once() {
    use super::super::{
        CompilerExecutionStartupInputV1 as Startup, OwnedExecutionInputs, dispatch,
    };
    use rustix::net::{AddressFamily, SocketFlags, SocketType, socketpair};
    use std::sync::Mutex;
    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, 4 * 1024 * 1024);
    b.reserve_storage(91).unwrap();
    b.charge_work(19).unwrap();
    let policy = valid_native_policy(&mut b);
    let (file, charge) = policy.try_clone_for_transfer(&mut b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (client, _server) = socketpair(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    let startup = Startup(Mutex::new(Some(Ok(OwnedExecutionInputs {
        policy: file.into(),
        service: client,
    }))));
    let ledger = b.work_ledger_identity_v1();
    let address = &b as *const Budget<'_> as usize;
    let floor = b.storage();
    {
        let selected = startup.admit_selected(&mut b).unwrap();
        let dispatch::Admitted::Native(Admitted {
            policy: retained,
            client,
            enrollment,
        }) = selected
        else {
            panic!("native image selected legacy");
        };
        assert_eq!(retained.policy(), policy.policy());
        assert!(enrollment.is_none());
        let (client, ()) = client
            .prepare::<_, Error>(|budget| {
                assert!(budget.work_ledger_identity_v1() == ledger);
                assert_eq!(budget as *const Budget<'_> as usize, address);
                assert!(budget.storage() > floor && budget.work() > 19);
                retained.revalidate(budget)?;
                budget.charge_work(7)?;
                Ok(())
            })
            .unwrap();
        drop(client);
        drop(enrollment);
    }
    assert!(b.work_ledger_identity_v1() == ledger && b.work() > 26);
    assert!(matches!(
        startup.admit_selected(&mut b),
        Err(Error::Startup(
            super::super::ProtectedCompilerExecutionErrorV1::InputAlreadyConsumed
        ))
    ));
    assert!(matches!(
        startup.admit(),
        Err(super::super::ProtectedCompilerExecutionErrorV1::InputAlreadyConsumed)
    ));
    policy.revalidate(&mut b).unwrap();
}

#[test]
fn native_session_refusal_closes_both_inputs_without_resetting_the_account() {
    if isolated(concat!(
        module_path!(),
        "::native_session_refusal_closes_both_inputs_without_resetting_the_account"
    )) {
        return;
    }
    for (prepaid, limit) in [
        (Admitted::INPUT_STORAGE - 1, usize::MAX),
        (Admitted::INPUT_STORAGE, 0),
        (Admitted::INPUT_STORAGE, usize::MAX),
    ] {
        let file = File::open("/dev/null").unwrap();
        for fd in [POLICY_FD, SERVICE_FD] {
            install(&file, fd, false);
        }
        let mut work = Work::new(limit);
        let mut b = Budget::new(&mut work, 4 * 1024 * 1024);
        b.reserve_storage(prepaid).unwrap();
        let ledger = b.work_ledger_identity_v1();
        // SAFETY: install relinquished the two live descriptors with into_raw_fd
        // in this isolated child; this attempt consumes that fresh transfer once.
        let result = unsafe { Admitted::admit(&mut b) };
        if prepaid < Admitted::INPUT_STORAGE {
            assert!(matches!(result, Err(Error::Resource(Resource::Accounting))));
        } else if limit == 0 {
            assert!(matches!(result, Err(Error::Resource(Resource::Work(_)))));
        } else {
            assert!(matches!(result, Err(Error::Policy(_))));
        }
        drop(result);
        assert_eq!(b.storage(), prepaid);
        assert!(b.work_ledger_identity_v1() == ledger);
        if limit == 0 {
            assert!(b.failed_work().is_some());
        }
        assert_closed();
    }
    missing_service_slot_cannot_become_a_private_policy_file();
}

fn valid_native_policy(b: &mut Budget<'_>) -> Policy {
    use fe2o3_compiler_execution_protocol::{
        CompilerExecutionIssuerMeasurementV1 as Measurement, CompilerExecutionIssuerPolicyV3,
    };
    // Two fixed valid public points for an inert sealed-policy fixture only.
    let mut key = [0x66; 32];
    key[0] = 0x58;
    let mut anchor = key;
    anchor[31] ^= 0x80;
    let (policy, charge) = CompilerExecutionIssuerPolicyV3::new(
        1,
        Measurement::new([1; 32], 1).unwrap(),
        Measurement::new([2; 32], 1).unwrap(),
        key,
        anchor,
        b,
    )
    .unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    let (policy, charge) = Policy::create(policy, b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    policy
}

fn missing_service_slot_cannot_become_a_private_policy_file() {
    let mut work = Work::new(usize::MAX);
    let mut b = Budget::new(&mut work, 4 * 1024 * 1024);
    let policy = valid_native_policy(&mut b);
    let (file, charge) = policy.try_clone_for_transfer(&mut b).unwrap();
    b.reserve_storage(charge.additional_storage()).unwrap();
    install(&file, POLICY_FD, false);
    let occupied = leave_service_hole(&file);
    b.reserve_storage(Admitted::INPUT_STORAGE).unwrap();
    let before = b.work();
    let floor = b.storage();
    // SAFETY: install relinquished the policy slot. The isolated fixture owns
    // every lower descriptor and keeps the service slot vacant through refusal.
    let result = unsafe { Admitted::admit(&mut b) };
    assert!(matches!(&result, Err(Error::Descriptor(e)) if e.raw_os_error() == Some(libc::EBADF)));
    drop(result);
    assert_eq!(b.work(), before + 2);
    assert_eq!(b.storage(), floor);
    assert_closed();
    policy.revalidate(&mut b).unwrap();
    drop(occupied);
}
