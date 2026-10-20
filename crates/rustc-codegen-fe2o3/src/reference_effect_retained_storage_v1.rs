//! Complete retained reference-effect binding payloads, not source authority.
use super::{
    AuthenticatedReferenceEffectBindingV1, AuthenticatedReferenceEffectBindingsV1,
    ReferenceBindingOriginV1, ReferenceEffectIrV1, ReferenceLogicalSignaturePreimageV1,
    ReferenceOutputWriteV1,
};
use fe2o3_kernel_ir::LogicalStorageCounterV1;
use fe2o3_verifier::portable_reference_v1::retained_storage_v1::ReferenceRetainedStorageErrorV1;

fn fixed<T: Copy>(_: &T) {}

impl AuthenticatedReferenceEffectBindingsV1 {
    /// Heap-only branch. Its inline Box handle belongs to the enclosing owner.
    /// Charge the actual binding Box, both live String capacities, both
    /// signature arrays, the complete effect IR and the separately owned deep
    /// output clone. The same caller Counter bounds every branch/node/allocation.
    /// First refusal stops; partial observation must be discarded. No producer,
    /// source authentication, canonical validation, proof or launch is granted.
    pub(crate) fn charge_retained_heap_storage_v1(
        &self,
        c: &mut LogicalStorageCounterV1,
    ) -> Result<(), ReferenceRetainedStorageErrorV1> {
        c.charge(0, 1)?;
        let Self { bindings } = self;
        let _: &Box<[AuthenticatedReferenceEffectBindingV1]> = bindings;
        c.array::<AuthenticatedReferenceEffectBindingV1>(bindings.len())?;
        for binding in bindings {
            c.charge(0, 1)?;
            let AuthenticatedReferenceEffectBindingV1 {
                origin,
                logical_kernel_name,
                kernel,
                reference,
                signature_preimage,
                effect_ir_sha256,
                effect_ir,
                observable_output_writes,
            } = binding;
            fixed(kernel);
            fixed(reference);
            fixed(effect_ir_sha256);
            match origin {
                ReferenceBindingOriginV1::SourceRegistration(path) => c.string(path)?,
                ReferenceBindingOriginV1::ReferenceEnrollment(origin) => fixed(origin),
            }
            charge_payload_without_registration(
                logical_kernel_name,
                signature_preimage,
                effect_ir,
                observable_output_writes,
                c,
            )?;
        }
        Ok(())
    }
}

/// Inert borrowed component view used by controls. No authenticated owner is
/// constructed. Its string/array handles and inline value headers already live
/// in one actual binding slot, which is charged only by the outer owner method.
fn charge_payload(
    registration_path: &String,
    logical_kernel_name: &String,
    signature: &ReferenceLogicalSignaturePreimageV1,
    ir: &ReferenceEffectIrV1,
    writes: &Box<[ReferenceOutputWriteV1]>,
    c: &mut LogicalStorageCounterV1,
) -> Result<(), ReferenceRetainedStorageErrorV1> {
    c.string(registration_path)?;
    charge_payload_without_registration(logical_kernel_name, signature, ir, writes, c)
}

fn charge_payload_without_registration(
    logical_kernel_name: &String,
    signature: &ReferenceLogicalSignaturePreimageV1,
    ir: &ReferenceEffectIrV1,
    writes: &Box<[ReferenceOutputWriteV1]>,
    c: &mut LogicalStorageCounterV1,
) -> Result<(), ReferenceRetainedStorageErrorV1> {
    c.string(logical_kernel_name)?;
    signature.charge_retained_heap_storage_v1(c)?;
    ir.charge_retained_heap_storage_v1(c)?;
    c.array::<ReferenceOutputWriteV1>(writes.len())?;
    for write in writes {
        write.charge_retained_heap_storage_v1(c)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "reference_effect_retained_storage_v1_tests.rs"]
mod tests;
