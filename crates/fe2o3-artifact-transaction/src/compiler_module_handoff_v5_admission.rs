//! Terminal conditional admission. Only successful transfers release scratch.
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1;
use std::convert::Infallible;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

/// Borrowable cause of a terminal failure. It is not a refundable error chain.
#[derive(Debug)]
pub enum CompilerModuleHandoffAdmissionCauseV5<E> {
    Transaction(Error),
    Admission(E),
}
type Cause<E> = CompilerModuleHandoffAdmissionCauseV5<E>;
impl<E> From<Error> for Cause<E> {
    fn from(error: Error) -> Self {
        Self::Transaction(error)
    }
}
impl<E> From<Resource> for Cause<E> {
    fn from(error: Resource) -> Self {
        Self::Transaction(error.into())
    }
}
impl<E> From<HandoffEngineError> for Cause<E> {
    fn from(error: HandoffEngineError) -> Self {
        Self::Transaction(error.into())
    }
}

/// Terminal failure retaining the lock until any callback-owned error is dropped.
/// No owned cause extraction is provided: `E` may own the moved handoff or a
/// partially recovered owner. Neither this wrapper nor its cause grants authority.
pub struct CompilerModuleHandoffAdmissionErrorV5<E> {
    // Declaration order is intentional: destroy all opaque custody under lock.
    cause: Cause<E>,
    _lock: crate::OutputLock,
}
impl<E> CompilerModuleHandoffAdmissionErrorV5<E> {
    pub const fn cause(&self) -> &Cause<E> {
        &self.cause
    }
}
impl<E: fmt::Debug> fmt::Debug for CompilerModuleHandoffAdmissionErrorV5<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CompilerModuleHandoffAdmissionErrorV5")
            .field(&self.cause)
            .finish()
    }
}
impl<E: fmt::Debug> fmt::Display for CompilerModuleHandoffAdmissionErrorV5<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "terminal conditional handoff admission: {:?}",
            self.cause
        )
    }
}
impl<E: fmt::Debug> std::error::Error for CompilerModuleHandoffAdmissionErrorV5<E> {}

// An arbitrary panic payload may also own recovery state. Preserve it and the
// lock together instead of releasing the lock before the catcher drops it.
struct LockedPanic {
    _payload: Box<dyn std::any::Any + Send>,
    _lock: crate::OutputLock,
}

struct Account {
    address: usize,
    ledger: CanonicalKernelIrWorkLedgerIdentityV1,
    floor: usize,
}
impl Account {
    fn new(budget: &Budget<'_>) -> Self {
        Self {
            address: budget as *const Budget<'_> as usize,
            ledger: budget.work_ledger_identity_v1(),
            floor: budget.storage(),
        }
    }
    fn check(&self, budget: &Budget<'_>, floor: usize) -> std::result::Result<(), Resource> {
        if self.address != budget as *const Budget<'_> as usize
            || self.ledger != budget.work_ledger_identity_v1()
            || budget.storage() < floor
        {
            return Err(Resource::Accounting);
        }
        Ok(())
    }
}

fn terminal_storage<T, E, F>() -> std::result::Result<usize, Resource> {
    FRAME_STORAGE
        .checked_add(size_of::<F>())
        .and_then(|n| n.checked_add(2 * size_of::<Account>()))
        .and_then(|n| n.checked_add(size_of::<LockedPanic>()))
        .and_then(|n| n.checked_add(size_of::<CompilerModuleHandoffAdmissionErrorV5<E>>()))
        .and_then(|n| n.checked_add(size_of::<std::result::Result<T, Cause<E>>>()))
        .and_then(|n| {
            n.checked_add(size_of::<
                std::thread::Result<std::result::Result<T, Cause<E>>>,
            >())
        })
        .ok_or(Resource::Arithmetic)
}

#[derive(Clone, Copy)]
enum TerminalAccount {
    Legacy,
    Original,
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_kernel_ir::{
        CanonicalKernelIrOwnedVerificationResourceBudgetV1 as Owned,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };

    #[test]
    fn original_terminal_exact_one_short_overflow_and_input_floor() {
        const OUTSIDE: usize = MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5 + 1;
        fn run(
            work: usize,
            limit: usize,
            inherited: usize,
            borrowed: usize,
        ) -> (bool, usize, usize) {
            let fixture = super::super::tests::fixture::Fixture::new();
            let output = PinnedOutput::open(&fixture.path).unwrap();
            // Lock acquisition is fixture setup, not a nonblocking-admission assertion.
            let lock = output.lock().unwrap();
            let mut owned = Owned::new(Work::new(work), limit);
            owned.with_budget(|b| {
                b.reserve_storage(OUTSIDE).unwrap();
                let identity = b.storage_account_identity_v1();
                let ledger = b.work_ledger_identity_v1();
                let result = terminal::<_, Infallible, _>(
                    b,
                    inherited,
                    borrowed,
                    lock,
                    TerminalAccount::Original,
                    |b| {
                        b.charge_work(17)?;
                        b.reserve_storage(37)?;
                        Ok(())
                    },
                );
                assert_eq!(b.storage_account_identity_v1(), identity);
                assert!(b.work_ledger_identity_v1() == ledger);
                assert_eq!(b.storage_limit(), limit);
                assert!(b.storage() >= OUTSIDE);
                if result.is_ok() {
                    assert_eq!(b.storage(), OUTSIDE);
                }
                (result.is_ok(), b.work(), b.peak_storage())
            })
        }
        let baseline = run(usize::MAX, OUTSIDE + 65536, 41, 43);
        assert!(baseline.0);
        assert_eq!(baseline.1, 8 + Budget::STORAGE_WINDOW_WORK_V1 + 17);
        assert_eq!(run(baseline.1, baseline.2, 41, 43), baseline);
        assert!(!run(baseline.1 - 1, baseline.2, 41, 43).0);
        assert!(!run(baseline.1, baseline.2 - 1, 41, 43).0);
        assert!(!run(baseline.1, baseline.2, OUTSIDE + 1, 0).0);
        assert!(!run(baseline.1, baseline.2, usize::MAX, 1).0);
    }

    #[test]
    fn terminal_fixture_waits_for_contended_output_lock() {
        let fixture = super::super::tests::fixture::Fixture::new();
        let output = PinnedOutput::open(&fixture.path).unwrap();
        let held = output.lock().unwrap();
        assert!(output.try_lock().unwrap().is_none());
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
        let (acquired_tx, acquired_rx) = std::sync::mpsc::sync_channel(0);
        std::thread::scope(|scope| {
            let output = &output;
            let contender = scope.spawn(move || {
                started_tx.send(()).unwrap();
                let _lock = output.lock().unwrap();
                acquired_tx.send(()).unwrap();
            });
            started_rx.recv().unwrap();
            assert!(matches!(
                acquired_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ));
            drop(held);
            acquired_rx.recv().unwrap();
            contender.join().unwrap();
        });
        let _reacquired = output.lock().unwrap();
    }

    #[test]
    fn conditional_transaction_v5_account_rejects_address_ledger_and_floor_substitution() {
        let mut work_a = Work::new(100);
        let mut work_b = Work::new(100);
        let mut a = Budget::new(&mut work_a, 100);
        let mut b = Budget::new(&mut work_b, 100);
        a.reserve_storage(37).unwrap();
        b.reserve_storage(37).unwrap();
        let original = Account::new(&a);
        original.check(&a, original.floor).unwrap();
        assert!(matches!(
            original.check(&b, original.floor),
            Err(Resource::Accounting)
        ));
        std::mem::swap(&mut a, &mut b);
        assert!(matches!(
            original.check(&a, original.floor),
            Err(Resource::Accounting)
        ));
        assert_eq!(a.storage(), 37);
        assert_eq!(b.storage(), 37);
        std::mem::swap(&mut a, &mut b);
        a.release_storage(1).unwrap();
        assert!(matches!(
            original.check(&a, original.floor),
            Err(Resource::Accounting)
        ));
        assert_eq!(a.storage(), 36);
    }
}

/// The lock lives outside the entire callback/postcheck/destructor catch. Unlike
/// Resources::scoped this releases nothing on refusal or unwind, even known
/// scratch: opaque partial recovery may still rely on the terminal reservation.
fn terminal<T, E, F>(
    budget: &mut Budget<'_>,
    inherited: usize,
    borrowed: usize,
    lock: crate::OutputLock,
    mode: TerminalAccount,
    f: F,
) -> std::result::Result<(T, crate::OutputLock), CompilerModuleHandoffAdmissionErrorV5<E>>
where
    F: FnOnce(&mut Budget<'_>) -> std::result::Result<T, Cause<E>>,
{
    let account = Account::new(budget);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let result = (|| {
            budget.charge_work(8)?;
            let inherited = inherited
                .checked_add(borrowed)
                .ok_or(Resource::Arithmetic)?;
            if account.floor < inherited {
                return Err(Resource::Accounting.into());
            }
            let run = |budget: &mut Budget<'_>| {
                budget.reserve_storage(terminal_storage::<T, E, F>()?)?;
                f(budget)
            };
            match mode {
                TerminalAccount::Legacy => {
                    if budget.storage_limit() > MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5 {
                        return Err(Resource::Accounting.into());
                    }
                    run(budget)
                }
                TerminalAccount::Original => {
                    // Count the full actual token AGAIN inside the local bound.
                    // The window itself refunds nothing on refusal or unwind.
                    let overlap = inherited
                        .checked_add(Budget::STORAGE_WINDOW_SCRATCH_V1)
                        .ok_or(Resource::Arithmetic)?;
                    budget.with_additional_storage_window_v1(
                        MAX_COMPILER_MODULE_HANDOFF_STORAGE_V5,
                        |budget| {
                            budget.reserve_storage(overlap)?;
                            run(budget)
                        },
                    )
                }
            }
        })();
        // A failed callback may retain arbitrary additional storage; only a
        // damaged original account/floor overrides that terminal cause here.
        account.check(budget, account.floor)?;
        if result.is_ok() {
            budget.release_storage(budget.storage() - account.floor)?;
        }
        result
    }));
    match result {
        Ok(Ok(value)) => Ok((value, lock)),
        Ok(Err(cause)) => Err(CompilerModuleHandoffAdmissionErrorV5 { cause, _lock: lock }),
        Err(payload) => resume_unwind(Box::new(LockedPanic {
            _payload: payload,
            _lock: lock,
        })),
    }
}

impl CompilerModuleHandoffConsumptionTokenV5 {
    /// Moves the exact transport through the caller's concrete recovery function.
    /// Success restores the callback checkpoint and returns additional unreserved
    /// storage. Reserve the returned receipt before using the mapped token.
    ///
    /// Errors may retain terminal storage and own the lock until dropped. Panic
    /// payloads also retain the lock until destroyed. No failure writes a consumed
    /// tombstone, returns a token or refunds work/storage. The generic callback
    /// does not itself establish compiler, proof, artifact or launch authority.
    pub fn try_map_handoff<T: AsRef<Handoff>, E>(
        self,
        budget: &mut Budget<'_>,
        admit: impl FnOnce(Handoff, &mut Budget<'_>) -> std::result::Result<(T, usize), E>,
    ) -> std::result::Result<
        (
            CompilerModuleHandoffConsumptionTokenV5<T>,
            CompilerModuleHandoffStorageV5,
        ),
        CompilerModuleHandoffAdmissionErrorV5<E>,
    > {
        self.try_map_handoff_using(budget, TerminalAccount::Legacy, admit)
    }

    /// Same terminal transfer on an ORIGINAL owned resource account. The full
    /// actual token charge is counted again within an additional <=256 MiB
    /// window; the original total ceiling, address and ledger never change.
    /// Callback refusal/unwind keeps ALL reservations and the actual output
    /// lock, just like the ordinary entry. This grants no admission authority.
    ///
    /// Additional work above the ordinary transfer is STORAGE_WINDOW_WORK_V1;
    /// additional scratch is the actual token storage plus
    /// STORAGE_WINDOW_SCRATCH_V1. Callback work/retained storage remain additional.
    pub fn try_map_handoff_in_original_account_v5<T: AsRef<Handoff>, E>(
        self,
        budget: &mut Budget<'_>,
        admit: impl FnOnce(Handoff, &mut Budget<'_>) -> std::result::Result<(T, usize), E>,
    ) -> std::result::Result<
        (
            CompilerModuleHandoffConsumptionTokenV5<T>,
            CompilerModuleHandoffStorageV5,
        ),
        CompilerModuleHandoffAdmissionErrorV5<E>,
    > {
        self.try_map_handoff_using(budget, TerminalAccount::Original, admit)
    }

    fn try_map_handoff_using<T: AsRef<Handoff>, E>(
        self,
        budget: &mut Budget<'_>,
        mode: TerminalAccount,
        admit: impl FnOnce(Handoff, &mut Budget<'_>) -> std::result::Result<(T, usize), E>,
    ) -> std::result::Result<
        (
            CompilerModuleHandoffConsumptionTokenV5<T>,
            CompilerModuleHandoffStorageV5,
        ),
        CompilerModuleHandoffAdmissionErrorV5<E>,
    > {
        let Self {
            binding,
            backing,
            content,
            storage,
            _lock,
        } = self;
        let ((content, additional, total), lock) =
            terminal(budget, storage.0, 0, _lock, mode, |budget| {
                let identity = content.identity();
                let headers = token_headers::<T>();
                budget.reserve_storage(headers)?;
                let account = Account::new(budget);
                let admitted = admit(content, budget);
                account.check(budget, account.floor)?;
                let admitted = match admitted {
                    Ok((owner, additional)) => {
                        if budget.storage() != account.floor {
                            return Err(Resource::Accounting.into());
                        }
                        // Inspect no opaque owner until its returned storage is paid.
                        budget.reserve_storage(additional)?;
                        let handoff = owner.as_ref();
                        if handoff.identity() != identity || backing_snapshot(handoff) != backing {
                            return Err(Error::HandoffIdentityMismatch.into());
                        }
                        let additional = additional
                            .checked_add(headers)
                            .ok_or(Resource::Arithmetic)?;
                        let total = storage
                            .0
                            .checked_add(additional)
                            .ok_or(Resource::Arithmetic)?;
                        Ok((owner, additional, total))
                    }
                    Err(error) => Err(Cause::Admission(error)),
                };
                // Stream scratch has a known local lifetime and starts above every
                // terminal callback reservation. It never encloses admission itself.
                currentness::stream(&binding, &mut Resources::Metered(budget))?;
                account.check(budget, account.floor)?;
                admitted
            })?;
        Ok((
            CompilerModuleHandoffConsumptionTokenV5 {
                binding,
                backing,
                content,
                storage: CompilerModuleHandoffStorageV5(total),
                _lock: lock,
            },
            CompilerModuleHandoffStorageV5(additional),
        ))
    }
}

pub(super) fn consume<T: AsRef<Handoff>>(
    lease: &CompilerModuleHandoffCurrentnessLeaseV5,
    token: CompilerModuleHandoffConsumptionTokenV5<T>,
    budget: &mut Budget<'_>,
    hooks: &mut impl HandoffHooks,
) -> Result<ConsumedCompilerModuleHandoffV5<T>> {
    consume_using(lease, token, budget, hooks, TerminalAccount::Legacy)
}

pub(super) fn consume_original<T: AsRef<Handoff>>(
    lease: &CompilerModuleHandoffCurrentnessLeaseV5,
    token: CompilerModuleHandoffConsumptionTokenV5<T>,
    budget: &mut Budget<'_>,
    hooks: &mut impl HandoffHooks,
) -> Result<ConsumedCompilerModuleHandoffV5<T>> {
    consume_using(lease, token, budget, hooks, TerminalAccount::Original)
}

fn consume_using<T: AsRef<Handoff>>(
    lease: &CompilerModuleHandoffCurrentnessLeaseV5,
    token: CompilerModuleHandoffConsumptionTokenV5<T>,
    budget: &mut Budget<'_>,
    hooks: &mut impl HandoffHooks,
    mode: TerminalAccount,
) -> Result<ConsumedCompilerModuleHandoffV5<T>> {
    let CompilerModuleHandoffConsumptionTokenV5 {
        binding,
        backing,
        content,
        storage,
        _lock,
    } = token;
    let borrowed = match mode {
        TerminalAccount::Legacy => 0,
        TerminalAccount::Original => lease.storage.0,
    };
    let result = terminal::<_, Infallible, _>(budget, storage.0, borrowed, _lock, mode, |budget| {
        if !Arc::ptr_eq(&lease.binding, &binding) {
            return Err(Error::MismatchedCurrentnessToken.into());
        }
        let handoff = content.as_ref();
        if handoff.identity() != binding.receipt.handoff_identity
            || backing_snapshot(handoff) != backing
        {
            return Err(Error::HandoffIdentityMismatch.into());
        }
        let resources = &mut Resources::Metered(budget);
        currentness::stream(&binding, resources)?;
        currentness::consume(&binding, resources, hooks)?;
        Ok(ConsumedCompilerModuleHandoffV5 {
            receipt: binding.receipt,
            content,
            storage,
        })
    });
    match result {
        Ok((consumed, _lock)) => Ok(consumed),
        Err(failure) => {
            // There is no opaque E in this operation. Its T has already been
            // destroyed inside terminal while the lock was held.
            let CompilerModuleHandoffAdmissionErrorV5 { cause, _lock } = failure;
            match cause {
                Cause::Transaction(error) => Err(error),
                Cause::Admission(never) => match never {},
            }
        }
    }
}
