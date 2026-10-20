//! Retained payloads of the actual transaction's two transcript owners.
//! This does not observe the earlier producer roster or authenticate new owners.
use super::{AuthenticatedRustcIdentityInventoryV3, AuthenticatedRustcPreflightPlanV3};

fn fixed<T: Copy>(_: &T) {}

impl AuthenticatedRustcIdentityInventoryV3 {
    /// Visits the transcript Box and optional original-root Vec backing.
    ///
    /// The transcript always has one visit; enrolled inventories add one actual
    /// root-capacity visit. The caller
    /// charges one callback visit and checked count*width bytes on its shared
    /// bounded ledger. Its enclosing owner pays this inline header/root once;
    /// this method does not pay either again. There is no variable traversal,
    /// allocation, hashing, cloning, validation, or authority conversion.
    ///
    /// The first callback error is returned unchanged. Discard an incomplete
    /// enclosing observation on error; accepted prefix charges are not undone.
    /// The Box extent is logical payload, not allocator overhead, scratch or RSS.
    pub(crate) fn visit_retained_heap_storage_v1<E>(
        &self,
        mut visit: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let Self {
            sha256,
            canonical_transcript,
            original_root_associations,
        } = self;
        fixed(sha256);
        // Do not silently use Vec length if the representation changes.
        let _: &Box<[u8]> = canonical_transcript;
        visit(canonical_transcript.len(), size_of::<u8>())?;
        if let Some(associations) = original_root_associations {
            associations.visit_retained_heap_storage_v1(visit)?;
        }
        Ok(())
    }
}

impl AuthenticatedRustcPreflightPlanV3 {
    /// One-Box, shared-ledger and header-exclusion contract. The inventory
    /// digest is inline, not another owner.
    /// No original source/function roster is retained by this wrapper.
    pub(crate) fn visit_retained_heap_storage_v1<E>(
        &self,
        mut visit: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let Self {
            sha256,
            rustc_identity_inventory_sha256,
            canonical_transcript,
        } = self;
        fixed(sha256);
        fixed(rustc_identity_inventory_sha256);
        let _: &Box<[u8]> = canonical_transcript;
        visit(canonical_transcript.len(), size_of::<u8>())
    }
}

#[cfg(test)]
#[path = "production_importer_transcript_storage_v1_tests.rs"]
mod tests;
