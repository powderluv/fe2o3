//! Service the compiler's existing issuer while the original runtime is running.
use super::*;

impl<T: Send + 'static> NativeAttempt<'_, T> {
    /// One bounded root-control poll. Failure is terminal for the enclosing
    /// Attempt, which owns cancellation and the original funded cleanup pool.
    pub(crate) fn service_publication(
        &mut self,
        cleanup: &mut Cleanup,
        maximum_handoff_bytes: usize,
        enrollment: &Option<fe2o3_compiler_lineage::NativeConditionalCpuMappingExpectationV1>,
        b: &mut Budget<'_>,
    ) -> Result<Storage> {
        let (growth, retained) = self.account.with(self.retained, b, |b| {
            let issuer = self
                .issuer
                .as_mut()
                .ok_or(Error::Invalid("root RPC lost original issuer"))?;
            let root = &mut self.root;
            let growth = self.trace.with_runtime_backing(b, |runtime, _, b| {
                issuer.child.with_resources(b, |payload, b| -> Result<_> {
                    Ok(root
                        .service_publication(
                            &mut issuer.connection,
                            runtime,
                            &issuer.child,
                            payload.prepared.trust.policy().policy(),
                            payload.manifest.manifest(),
                            cleanup,
                            maximum_handoff_bytes,
                            enrollment,
                            b,
                        )?
                        .additional_storage())
                })
            })?;
            b.reserve_storage(growth)?;
            Ok((growth, sum(&[self.retained, growth])?))
        })?;
        self.retained = retained;
        Ok(Storage(growth))
    }

    /// Require the original Prepare-time occurrence's durable retirement. The
    /// sole late slot is never reacquired at exit or after consuming terminal wait.
    pub(crate) fn require_retired_publication(&mut self, b: &mut Budget<'_>) -> Result<()> {
        self.account.with(self.retained, b, |b| {
            let root = &self.root;
            self.trace
                .with_runtime_backing(b, |runtime, _, b| -> Result<()> {
                    root.retired_publication_subject(runtime, b)?;
                    Ok(())
                })
        })
    }

    pub(crate) fn publication_service_quota(maximum_handoff_bytes: usize) -> Result<Quota> {
        let access = Self::runtime_backing_quota()?;
        let rpc = RootSession::publication_service_quota(
            crate::native_v3::MAX_LAUNCHER,
            maximum_handoff_bytes,
        )?;
        Ok(Quota {
            work: sum(&[
                access.work(),
                Resources::<Payload<T>>::ACCESS_WORK,
                rpc.work(),
            ])?,
            scratch: sum(&[
                access.scratch(),
                Resources::<Payload<T>>::ACCESS_SCRATCH,
                rpc.scratch(),
            ])?,
        })
    }

    pub(crate) fn retired_publication_quota() -> Result<Quota> {
        let access = Self::runtime_backing_quota()?;
        let subject = RootSession::retired_publication_subject_quota();
        Ok(Quota {
            work: sum(&[access.work(), subject.work()])?,
            scratch: sum(&[access.scratch(), subject.scratch()])?,
        })
    }
}
