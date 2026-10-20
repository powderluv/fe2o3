//! Inventory coordinates from a signed, independently policy-pinned input.
//! This is content validation, not fresh currentness or mapped source recovery.
use super::*;
use fe2o3_compiler_lineage::{
    RUSTC_ENROLLMENT_INVENTORY_WORKING_STORAGE_V1 as INVENTORY_SCRATCH,
    RUSTC_IDENTITY_INVENTORY_PREFIX_WORK_V2 as LEGACY_WORK,
    RustcEnrollmentInventoryErrorV1 as InventoryError,
    has_rustc_identity_inventory_prefix_v2 as is_legacy,
    read_rustc_enrollment_inventory_v1 as read_inventory,
};
use fe2o3_rustc_invocation::{
    MAX_REFERENCE_ENROLLMENT_BINDINGS_V1, ReferenceEnrollmentDecodeErrorV1 as DecodeError,
    ReferenceEnrollmentRequestV1 as Enrollment,
};

const ENTRY: usize = 8;
pub(super) const SCRATCH: usize = INVENTORY_SCRATCH
    + size_of::<(
        Expected,
        Option<Expected>,
        Option<usize>,
        Enrollment,
        Sha256,
    )>()
    + 128;

impl AuthenticatedInput<'_> {
    // authenticate first verifies the carriage signature against its separately
    // pinned policy and joins the raw Subject to this exact retained handoff.
    // No caller inventory/header or live process environment supplies coordinates.
    pub(super) fn require_enrollment(&mut self, policy: &Policy, b: &mut Budget<'_>) -> Result<()> {
        require_original_account(self.ledger, self.account, self.floor, b)?;
        b.with_prepaid_scope(
            self.floor,
            ENTRY,
            ENTRY + Expected::HEADER_MATCH_WORK,
            SCRATCH,
            |b| {
                let capsule = self.handoff.capsule();
                let count = Enrollment::project_binding_count_from_descriptor(
                    capsule.invocation(),
                    |work| b.charge_work(work),
                )
                .map_err(|error| match error {
                    DecodeError::Work(error) => Error::Resource(error),
                    _ => failure("native captured enrollment request"),
                })?;
                // None is original absence; a present empty request never falls
                // back to registration-only inventory.
                let count = match count {
                    None => {
                        b.charge_work(LEGACY_WORK)?;
                        require(
                            is_legacy(capsule.rustc_identity_inventory().canonical_preimage()),
                            "native absent enrollment requires original legacy inventory",
                        )?;
                        return Ok(());
                    }
                    Some(count) if (1..=MAX_REFERENCE_ENROLLMENT_BINDINGS_V1).contains(&count) => {
                        u32::try_from(count).map_err(|_| Resource::Arithmetic)?
                    }
                    Some(_) => return Err(failure("native captured enrollment count")),
                };
                let expected = Expected {
                    // The signed Subject digest uses the V3 domain. Inventory
                    // coordinates use raw SHA-256 of these already joined bytes.
                    rustc_invocation_sha256: {
                        let invocation = capsule.invocation_bytes();
                        b.charge_work(
                            invocation
                                .len()
                                .checked_add(128)
                                .ok_or(Resource::Arithmetic)?,
                        )?;
                        Sha256::digest(invocation).into()
                    },
                    native_policy_sha256: *policy.identity().as_bytes(),
                    policy_generation: policy.generation(),
                    enrollment_binding_count: count,
                };
                let inventory = read_inventory(
                    capsule.rustc_identity_inventory().canonical_preimage(),
                    INVENTORY_SCRATCH,
                    |work| b.charge_work(work),
                )
                .map_err(|error| match error {
                    InventoryError::Charge(error) => Error::Resource(error),
                    _ => failure("native retained enrollment inventory"),
                })?;
                require(
                    expected.matches_header(&inventory.header()),
                    "native enrollment inventory differs from signed original invocation",
                )?;
                // Matching coordinates do not compare root Instances or CPU/source
                // semantics. The actual mapped verifier remains a separate gate.
                require_original_account(self.ledger, self.account, self.floor, b)?;
                self.enrollment = Some(expected);
                Ok(())
            },
        )?;
        require_original_account(self.ledger, self.account, self.floor, b)
    }
}
