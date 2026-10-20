//! Private original-Instance capture, sealed before the preflight inventory hash.
use super::{
    AuthenticatedReferenceEffectBindingsV1, CollectedFunction, CollectedFunctionRole,
    ReferenceBindingErrorV1 as Error, ReferenceBindingOriginV1, RetainedReferenceInputsV1,
    SourceClosureWorkV1 as Work, charge, error,
};
use crate::protected_compiler_execution::native_v3::{
    OriginalEnrollmentInventoryContextV1 as Context, ReferenceEnrollmentLoanV1 as Loan,
    RetainedReferenceEnrollmentStampV1,
};
use crate::rustc_semantic_adapter_v1::canonical_function_identities_v1;
use crate::rustc_semantic_plan_v1::{RetainedSemanticFunctionProducerV1, SourceClosureWorkStampV1};
use fe2o3_compiler_lineage::{
    RustcEnrollmentInventoryHeaderV1 as Header, RustcEnrollmentInventoryRootV1 as Row,
};
use fe2o3_mir_model::semantic_mir_v1::{HARD_MAX_ROOTS_V1, SemanticFunctionIdV1};
use rustc_middle::ty::{Instance, TyCtxt};
use sha2::{Digest, Sha256};
use std::mem::size_of;

struct PendingRootV1<'tcx> {
    kernel: Instance<'tcx>,
    reference: Instance<'tcx>,
    origin_tag: u8,
    descriptor_ordinal: u32,
    logical_name_len: u32,
    logical_name_sha256: [u8; 32],
    kernel_binding: [u8; 32],
}

pub(in crate::collector) struct PendingOriginalRootAssociationsV1<'tcx> {
    session: &'tcx rustc_session::Session,
    source: SourceClosureWorkStampV1,
    issuer: RetainedReferenceEnrollmentStampV1,
    context: Context,
    roots: Vec<PendingRootV1<'tcx>>,
}

/// Only original rederivation and the canonical importer can construct this owner.
#[derive(Debug)]
pub(crate) struct RetainedOriginalRootAssociationsV1 {
    header: Header,
    roots: Vec<Row>,
}

impl RetainedOriginalRootAssociationsV1 {
    #[cfg(test)]
    pub(in crate::collector) fn check_paid_spare_capacity_for_test(&mut self, loan: &Loan<'_>) {
        let before = self.roots.capacity();
        let additional = before + 4;
        loan.reserve_inventory_storage(additional * size_of::<Row>())
            .unwrap();
        self.roots.try_reserve_exact(additional).unwrap();
        let actual = self.roots.capacity();
        loan.reserve_inventory_storage(
            actual.saturating_sub(before + additional) * size_of::<Row>(),
        )
        .unwrap();
        assert!(actual > self.roots.len());
        let mut visits = Vec::new();
        self.visit_retained_heap_storage_v1(|count, size| {
            visits.push((count, size));
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
        assert_eq!(visits, [(actual, size_of::<Row>())]);
    }

    pub(crate) fn header(&self) -> Header {
        self.header
    }

    pub(crate) fn roots(&self) -> &[Row] {
        &self.roots
    }

    pub(crate) fn visit_retained_heap_storage_v1<E>(
        &self,
        mut visit: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        visit(self.roots.capacity(), size_of::<Row>())
    }
}

fn paid_vector<T>(count: usize, loan: &Loan<'_>, work: &mut Work) -> Result<Vec<T>, Error> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| error("root mapping allocation overflow"))?;
    charge(
        work,
        bytes
            .checked_add(1)
            .ok_or_else(|| error("root mapping work overflow"))?,
    )?;
    loan.reserve_inventory_storage(
        bytes
            .checked_add(size_of::<Vec<T>>())
            .ok_or_else(|| error("root mapping header overflow"))?,
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| error("root mapping allocation failed"))?;
    let actual = values
        .capacity()
        .checked_mul(size_of::<T>())
        .ok_or_else(|| error("root mapping capacity overflow"))?;
    loan.reserve_inventory_storage(
        actual
            .checked_sub(bytes)
            .ok_or_else(|| error("root mapping capacity accounting"))?,
    )?;
    Ok(values)
}

impl<'tcx> RetainedReferenceInputsV1<'tcx> {
    pub(in crate::collector) fn rederive_with_inventory_capture(
        &self,
        tcx: TyCtxt<'tcx>,
        functions: &[CollectedFunction<'tcx>],
        work: &mut Work,
        loan: Option<&Loan<'_>>,
    ) -> Result<
        (
            AuthenticatedReferenceEffectBindingsV1,
            Option<PendingOriginalRootAssociationsV1<'tcx>>,
        ),
        Error,
    > {
        let bindings = self.rederive_with_enrollment(tcx, functions, work, loan)?;
        let Some(loan) = loan else {
            return Ok((bindings, None));
        };
        let Some(context) = loan.inventory_context(work)? else {
            return Ok((bindings, None));
        };
        let enrollment = self
            .enrollment
            .as_ref()
            .ok_or_else(|| error("root mapping has no original enrollment owner"))?;
        enrollment.revalidate(tcx, loan, work)?;
        enrollment.validate_inputs(functions, work)?;
        charge(work, functions.len())?;
        let kernel_count = functions
            .iter()
            .filter(|function| function.role == CollectedFunctionRole::KernelEntry)
            .count();
        if kernel_count as u64 > HARD_MAX_ROOTS_V1
            || kernel_count != self.inputs.len()
            || bindings.as_slice().len() != self.inputs.len()
        {
            return Err(error(
                "root mapping requires every original KernelEntry reference",
            ));
        }
        charge(work, size_of::<PendingOriginalRootAssociationsV1<'tcx>>())?;
        loan.reserve_inventory_storage(size_of::<PendingOriginalRootAssociationsV1<'tcx>>())?;
        let source = work
            .retain_identity_v1()
            .map_err(|failure| Error::new(failure.to_string()))?;
        let issuer = loan.capture_stamp(work)?;
        let mut roots = paid_vector(self.inputs.len(), loan, work)?;
        for (input, fresh) in self.inputs.iter().zip(bindings.as_slice()) {
            charge(work, 1)?;
            let function = functions
                .get(input.function)
                .ok_or_else(|| error("root mapping original function missing"))?;
            if function.role != CollectedFunctionRole::KernelEntry
                || function.instance != input.kernel
                || function.reference_instance != Some(input.reference)
                || fresh.logical_kernel_name != input.logical_kernel_name
                || fresh.origin != input.origin
            {
                return Err(error("root mapping original pair changed"));
            }
            let name = input.logical_kernel_name.as_bytes();
            charge(
                work,
                name.len()
                    .checked_add(size_of::<PendingRootV1<'tcx>>())
                    .ok_or_else(|| error("root mapping name work overflow"))?,
            )?;
            let (origin_tag, descriptor_ordinal) = match &input.origin {
                ReferenceBindingOriginV1::SourceRegistration(_) => (0, 0),
                ReferenceBindingOriginV1::ReferenceEnrollment(origin) => {
                    if origin.rustc_invocation_sha256 != context.rustc_invocation_sha256
                        || origin.native_policy_sha256 != context.native_policy_sha256
                        || origin.policy_generation != context.policy_generation
                        || origin.mapping_ordinal >= context.descriptor_bindings
                        || loan.origin(origin.mapping_ordinal)? != *origin
                    {
                        return Err(error("root mapping original descriptor origin changed"));
                    }
                    (1, origin.mapping_ordinal)
                }
            };
            roots.push(PendingRootV1 {
                kernel: input.kernel,
                reference: input.reference,
                origin_tag,
                descriptor_ordinal,
                logical_name_len: u32::try_from(name.len())
                    .map_err(|_| error("root mapping name length overflow"))?,
                logical_name_sha256: Sha256::digest(name).into(),
                kernel_binding: function
                    .kernel_binding
                    .ok_or_else(|| error("root mapping kernel binding missing"))?
                    .as_bytes(),
            });
        }
        enrollment.revalidate(tcx, loan, work)?;
        Ok((
            bindings,
            Some(PendingOriginalRootAssociationsV1 {
                session: tcx.sess,
                source,
                issuer,
                context,
                roots,
            }),
        ))
    }
}

impl<'tcx> PendingOriginalRootAssociationsV1<'tcx> {
    pub(in crate::collector) fn seal(
        self,
        tcx: TyCtxt<'tcx>,
        loan: &Loan<'_>,
        work: &mut Work,
        functions: &[RetainedSemanticFunctionProducerV1<'tcx>],
        canonical_roots: &[SemanticFunctionIdV1],
    ) -> Result<RetainedOriginalRootAssociationsV1, Error> {
        if !std::ptr::eq(self.session, tcx.sess)
            || !work
                .matches_identity_v1(&self.source)
                .map_err(|failure| Error::new(failure.to_string()))?
        {
            return Err(error(
                "root mapping original Session or SOURCE account changed",
            ));
        }
        loan.revalidate_stamp(&self.issuer, work)?;
        if loan.inventory_context(work)? != Some(self.context) {
            return Err(error("root mapping original issuer header changed"));
        }
        loan.reserve_inventory_storage(size_of::<RetainedOriginalRootAssociationsV1>())?;
        let mut rows: Vec<Row> = paid_vector(self.roots.len(), loan, work)?;
        for root in canonical_roots {
            charge(work, 1)?;
            let function = functions
                .get(root.index() as usize)
                .ok_or_else(|| error("root mapping semantic root missing"))?;
            if function.role == CollectedFunctionRole::DeviceFfiExport {
                continue;
            }
            if function.role != CollectedFunctionRole::KernelEntry {
                return Err(error("root mapping semantic root role changed"));
            }
            let mut selected = None;
            for candidate in &self.roots {
                charge(work, 1)?;
                if candidate.kernel == function.instance && selected.replace(candidate).is_some() {
                    return Err(error("root mapping original kernel is ambiguous"));
                }
            }
            let selected =
                selected.ok_or_else(|| error("root mapping original kernel is missing"))?;
            if function.kernel_binding.map(|binding| binding.as_bytes())
                != Some(selected.kernel_binding)
            {
                return Err(error("root mapping canonical kernel binding changed"));
            }
            charge(
                work,
                selected
                    .reference
                    .args
                    .len()
                    .checked_mul(32)
                    .and_then(|n| n.checked_add(160))
                    .ok_or_else(|| error("root mapping identity work overflow"))?,
            )?;
            let reference = canonical_function_identities_v1(tcx, selected.reference).function();
            charge(
                work,
                rows.len()
                    .checked_mul(size_of::<Row>())
                    .ok_or_else(|| error("root mapping census work overflow"))?,
            )?;
            if rows.iter().any(|row| {
                row.semantic_root >= root.index()
                    || row.kernel_instance == *function.identities.function().as_bytes()
                    || (selected.origin_tag == 1
                        && row.origin_tag == 1
                        && row.descriptor_ordinal == selected.descriptor_ordinal)
            }) {
                return Err(error("root mapping canonical root or ordinal duplicate"));
            }
            rows.push(Row {
                semantic_root: root.index(),
                origin_tag: selected.origin_tag,
                descriptor_ordinal: selected.descriptor_ordinal,
                logical_name_len: selected.logical_name_len,
                logical_name_sha256: selected.logical_name_sha256,
                kernel_binding: selected.kernel_binding,
                kernel_instance: *function.identities.function().as_bytes(),
                reference_instance: *reference.as_bytes(),
            });
        }
        charge(work, rows.len())?;
        if rows.len() != self.roots.len()
            || rows.iter().filter(|row| row.origin_tag == 1).count()
                != self.context.descriptor_bindings as usize
        {
            return Err(error("root mapping incomplete kernel or descriptor census"));
        }
        loan.revalidate_stamp(&self.issuer, work)?;
        if !work
            .matches_identity_v1(&self.source)
            .map_err(|failure| Error::new(failure.to_string()))?
        {
            return Err(error(
                "root mapping original SOURCE account changed after sealing",
            ));
        }
        Ok(RetainedOriginalRootAssociationsV1 {
            header: Header {
                kernel_count: u32::try_from(rows.len())
                    .map_err(|_| error("root mapping count overflow"))?,
                enrollment_binding_count: self.context.descriptor_bindings,
                invocation_identity: self.context.rustc_invocation_sha256,
                native_policy_identity: self.context.native_policy_sha256,
                native_policy_generation: self.context.policy_generation,
            },
            roots: rows,
        })
    }
}
