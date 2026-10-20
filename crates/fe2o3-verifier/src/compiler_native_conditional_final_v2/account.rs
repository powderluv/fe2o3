use super::*;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

pub(crate) struct Account {
    address: usize,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    floor: usize,
}
impl Account {
    pub(crate) fn capture(budget: &Budget<'_>) -> Self {
        Self {
            address: budget as *const Budget<'_> as usize,
            ledger: budget.work_ledger_identity_v1(),
            floor: budget.storage(),
        }
    }
    pub(crate) fn require(&self, budget: &Budget<'_>, owned: usize) -> Result<(), Error> {
        if self.address != budget as *const Budget<'_> as usize
            || self.ledger != budget.work_ledger_identity_v1()
            || self
                .floor
                .checked_add(owned)
                .is_none_or(|floor| budget.storage() < floor)
        {
            return Err(Resource::Accounting.into());
        }
        Ok(())
    }

    pub(crate) fn require_exact(&self, budget: &Budget<'_>, owned: usize) -> Result<(), Error> {
        self.require(budget, owned)?;
        if self.floor.checked_add(owned) != Some(budget.storage()) {
            return Err(Resource::Accounting.into());
        }
        Ok(())
    }
}

/// Closed-route temporary reservation. A failed or unwound nested operation
/// leaves terminal charges; only denial-free success releases this known extent.
pub(super) fn temporary<T>(
    budget: &mut Budget<'_>,
    bytes: usize,
    run: impl FnOnce(&mut Budget<'_>) -> Result<T, Error>,
) -> Result<T, Error> {
    temporary_using(budget, bytes, run)
}

pub(crate) fn temporary_using<T, Failure: From<Error> + From<Resource>>(
    budget: &mut Budget<'_>,
    bytes: usize,
    run: impl FnOnce(&mut Budget<'_>) -> Result<T, Failure>,
) -> Result<T, Failure> {
    budget.check_prior_denials_v1()?;
    let account = Account::capture(budget);
    budget.charge_work(1)?;
    budget.reserve_storage(bytes)?;
    let result = run(budget);
    if let Err(error) = account.require(budget, bytes) {
        drop(result);
        return Err(error.into());
    }
    if result.is_ok() {
        budget.check_prior_denials_v1()?;
        budget.release_storage(bytes)?;
    }
    result
}

/// Existing C1 custody is unreserved only after the entire closed route succeeds.
/// Never invoke source-only C1 refund classification on a final-route error.
#[cfg(test)]
pub(super) fn transfer<T>(
    budget: &mut Budget<'_>,
    run: impl FnOnce(&mut Budget<'_>) -> Result<(T, NativeConditionalSourceStorageV2), Error>,
) -> Result<(T, NativeConditionalSourceStorageV2), Error> {
    transfer_using(budget, run)
}

pub(super) fn transfer_using<T, Failure: From<Error> + From<Resource>>(
    budget: &mut Budget<'_>,
    run: impl FnOnce(&mut Budget<'_>) -> Result<(T, NativeConditionalSourceStorageV2), Failure>,
) -> Result<(T, NativeConditionalSourceStorageV2), Failure> {
    budget.check_prior_denials_v1()?;
    let account = Account::capture(budget);
    budget.charge_work(8)?;
    let result = catch_unwind(AssertUnwindSafe(|| run(budget)));
    let result = match result {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    };
    if let Err(error) = account.require(budget, 0) {
        drop(result);
        return Err(error.into());
    }
    match result {
        Ok((owner, storage)) => {
            if let Err(error) = account.require(budget, storage.retained_storage()) {
                drop(owner);
                return Err(error.into());
            }
            budget.check_prior_denials_v1()?;
            if let Err(error) = budget.release_storage(budget.storage() - account.floor) {
                drop(owner);
                return Err(error.into());
            }
            Ok((owner, storage))
        }
        Err(error) => Err(error),
    }
}
