//! A monotone pair of actual artifact owners, never an observation/authority cache.
use super::*;

pub(super) struct Owners {
    resources: CustodyResources,
    started: bool,
    publication: Option<Lease>,
    token: Option<Token>,
}

impl Owners {
    pub(super) fn new(quote: Quote, b: &mut Budget<'_>) -> Result<Self> {
        Ok(Self {
            resources: CustodyResources::prepare(quote, b).map_err(NativeOccurrenceError::from)?,
            started: false,
            publication: None,
            token: None,
        })
    }

    fn acquire(
        &mut self,
        output: &std::path::Path,
        producer: &fe2o3_artifact_transaction::ProducerIdentity,
        receipt: Receipt,
        b: &mut Budget<'_>,
    ) -> Result<()> {
        if self.started || self.publication.is_some() || self.token.is_some() {
            return Err(RootPublicationCustodyErrorV3::state(
                "publication acquisition already started",
            ));
        }
        if receipt != self.resources.quote().receipt() {
            return Err(RootPublicationCustodyErrorV3::state(
                "publication quote changed receipt",
            ));
        }
        self.started = true;
        let barrier = retirement_barrier()?;
        let Self {
            resources,
            publication,
            token,
            ..
        } = self;
        resources
            .with_acquisition(&barrier, b, |scope| {
                let (lease, storage) = scope.acquire_lease(output, producer)?;
                *publication = Some(lease);
                scope.reserve_retained(storage)?;
                let lease = publication.as_ref().expect("installed publication lease");
                let (current, storage) = scope.acquire_token(lease)?;
                *token = Some(current);
                scope.reserve_retained(storage)?;
                Ok(())
            })
            .map_err(NativeOccurrenceError::from)?;
        Ok(())
    }

    fn current(&self) -> Result<(&Lease, &Token)> {
        match (&self.publication, &self.token) {
            (Some(publication), Some(token)) => Ok((publication, token)),
            _ => Err(RootPublicationCustodyErrorV3::state(
                "publication custody is incomplete",
            )),
        }
    }
}

// SAFETY: the payload owns only concrete Send artifact types. A successful
// barrier excludes every in-progress/new spawn until BOTH owners have dropped.
// Busy or dropped preparation changes no ownership. No budget or I/O validation
// occurs in retire; the fixed bound covers maximal decoded backing destruction.
#[allow(unsafe_code)]
unsafe impl Payload for Owners {
    const RETIRE_WORK: usize = 64 * 1088;
    const RETIRE_WORK_PER_BYTE: usize = 64;
    const RETIRE_SCRATCH: usize = 4 * size_of::<(Self, Barrier)>() + 4096;
    type Prepared = Barrier;
    fn try_prepare_retirement(&self) -> Option<Barrier> {
        retirement_barrier().ok()
    }
    fn retire(self, prepared: Barrier) {
        drop(self);
        drop(prepared);
    }
}

pub(super) struct Acquire<'a, 'trace, 'work> {
    pub root: &'a RootObservation<'trace, 'work>,
    pub observation: &'a NativeObservation,
    pub expected: &'a Expected,
    pub receipt: Receipt,
    pub output: &'a mut Option<(Subject, [u8; 32])>,
}

// SAFETY: only this exact, prequoted receipt can be acquired; each returned owner
// moves into its sole slot BEFORE any fallible accounting/validation. The barrier
// covers constructor rollback. Errors/unwind preserve every installed owner.
#[allow(unsafe_code)]
unsafe impl Build<Acquire<'_, '_, '_>> for Owners {
    const BUILD_WORK: usize = OBSERVE_WORK;
    const BUILD_SCRATCH: usize = FRAME;
    type Error = RootPublicationCustodyErrorV3;
    fn build(&mut self, op: Acquire<'_, '_, '_>, b: &mut Budget<'_>) -> Result<()> {
        op.observation.revalidate_from(Source::Root(op.root), b)?;
        let output = op.observation.output_dir();
        self.acquire(&output, &op.expected.producer, op.receipt, b)?;
        let (publication, token) = self.current()?;
        publication
            .validate_current_token(token)
            .map_err(NativeOccurrenceError::from)?;
        require_joined_invocation(op.observation, token)?;
        let (subject, storage) = self
            .resources
            .subject(publication, token, b)
            .map_err(NativeOccurrenceError::from)?;
        b.reserve_storage(storage.retained_storage())?;
        let observed = finish_join_subject(op.observation, op.expected, subject)?;
        self.resources
            .revalidate(publication, token, b)
            .map_err(NativeOccurrenceError::from)?;
        op.observation.revalidate_from(Source::Root(op.root), b)?;
        *op.output = Some(observed);
        Ok(())
    }
}

pub(super) struct Validate<'a, 'trace, 'work> {
    pub root: &'a RootObservation<'trace, 'work>,
    pub observation: &'a NativeObservation,
    pub enrollment: Option<&'a Option<Enrollment>>,
}

// SAFETY: this operation never mutates the concrete payload or exports a view.
#[allow(unsafe_code)]
unsafe impl Build<Validate<'_, '_, '_>> for Owners {
    const BUILD_WORK: usize = LOCAL_WORK;
    const BUILD_SCRATCH: usize = FRAME;
    type Error = RootPublicationCustodyErrorV3;
    fn build(&mut self, op: Validate<'_, '_, '_>, b: &mut Budget<'_>) -> Result<()> {
        let (publication, token) = self.current()?;
        publication
            .validate_current_token(token)
            .map_err(NativeOccurrenceError::from)?;
        op.observation.revalidate_from(Source::Root(op.root), b)?;
        self.resources
            .revalidate(publication, token, b)
            .map_err(NativeOccurrenceError::from)?;
        if let Some(expected) = op.enrollment {
            // The original quote covers the full token capacity and metadata,
            // not just this borrowed inventory's visible length.
            enrollment::require(
                token
                    .handoff()
                    .capsule()
                    .rustc_identity_inventory()
                    .canonical_preimage(),
                self.resources.quote().retained_storage(),
                expected,
                b,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "compiler_execution_root_publication_payload_tests.rs"]
mod tests;
