//! Independent fixed-policy framing reuses the canonical source-bound roster.
use fe2o3_compiler_lineage::{
    MAX_NATIVE_CONDITIONAL_STORAGE_V1, NATIVE_CONDITIONAL_POLICY_ROSTER_WORKING_STORAGE_V1,
    NativeConditionalPolicyRosterRefV1 as Roster, read_native_conditional_policy_roster_v1,
};
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as ResourceError,
};
use sha2::{Digest, Sha256};
use std::io;

pub const MAX_NATIVE_CONDITIONAL_ROOT_POLICY_FILE_BYTES_V1: usize = 131_688;
const MAX_POLICY_BYTES: usize = MAX_NATIVE_CONDITIONAL_ROOT_POLICY_FILE_BYTES_V1;
const MIN_BYTES: usize = 648;
const HEADER: usize = 72;
const MAGIC: &[u8; 8] = b"F3NCRP1\0";
const DOMAIN: &[u8] = b"FE2O3/NATIVE-CONDITIONAL-ROOT-POLICY-FILE/V1\0";
const SCRATCH: usize = NATIVE_CONDITIONAL_POLICY_ROSTER_WORKING_STORAGE_V1 + 4096;

struct View<'a> {
    roster: Roster<'a>,
}

/// Inert format validation only. This neither accepts the roster nor authenticates
/// the installed file. The fixed native deployment must independently pin its
/// raw SHA-256 and exact byte length and retain the original protected file.
pub fn validate_native_conditional_root_policy_file_v1(
    bytes: &[u8],
    budget: &mut Budget<'_>,
) -> io::Result<()> {
    read(bytes, budget).map(|_| ())
}

/// Wraps an already encoded canonical roster for explicit administrative
/// provisioning. Only singleton gfx942:xnack-/wave64 conditional fill, production
/// history limits V1 and final-KIR-to-machine boundary 6 are selected. Acceptance
/// never follows from constructing these bytes or from a source-supplied roster.
/// Returns the full unreserved Vec/header charge; input backing remains prepaid.
pub fn encode_native_conditional_root_policy_file_v1(
    roster: &[u8],
    budget: &mut Budget<'_>,
) -> io::Result<(Vec<u8>, usize)> {
    let length = roster
        .len()
        .checked_add(HEADER + 32)
        .ok_or_else(|| io::Error::other("native policy extent overflow"))?;
    budget.charge_work(8).map_err(other)?;
    require(
        (MIN_BYTES..=MAX_POLICY_BYTES).contains(&length),
        "native policy extent",
    )?;
    let work = length
        .checked_mul(4)
        .ok_or_else(|| io::Error::other("native policy work overflow"))?;
    refundable_scope(budget, roster.len(), work, SCRATCH + length, |b| {
        let decoded = read_roster(roster, b)?;
        require(
            decoded.root_count() == 1,
            "native fill policy requires one root",
        )?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(other)?;
        require(
            bytes.capacity() == length,
            "native policy allocation capacity",
        )?;
        bytes.resize(length, 0);
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
        bytes[12..16].copy_from_slice(&(length as u32).to_le_bytes());
        bytes[24..26].copy_from_slice(&1u16.to_le_bytes()); // gfx942:xnack-, wave64
        bytes[26..28].copy_from_slice(&1u16.to_le_bytes()); // production history V1
        bytes[28..30].copy_from_slice(&6u16.to_le_bytes());
        bytes[32..40].copy_from_slice(&(roster.len() as u64).to_le_bytes());
        bytes[40..72].copy_from_slice(&Sha256::digest(roster));
        bytes[HEADER..length - 32].copy_from_slice(roster);
        let hash = checksum(&bytes[..length - 32]);
        bytes[length - 32..].copy_from_slice(&hash);
        Ok((bytes, length + size_of::<Vec<u8>>() + size_of::<usize>()))
    })
}

fn read<'a>(bytes: &'a [u8], budget: &mut Budget<'_>) -> io::Result<View<'a>> {
    budget.check_prior_denials_v1().map_err(other)?;
    budget.charge_work(8).map_err(other)?;
    require(
        (MIN_BYTES..=MAX_POLICY_BYTES).contains(&bytes.len()),
        "native policy extent",
    )?;
    let work = bytes
        .len()
        .checked_mul(4)
        .ok_or_else(|| io::Error::other("native policy work overflow"))?;
    refundable_scope(budget, bytes.len(), work, SCRATCH, |b| {
        require(
            &bytes[..8] == MAGIC
                && bytes[8..10] == 1u16.to_le_bytes()
                && bytes[10..12] == [0; 2]
                && bytes[12..16] == (bytes.len() as u32).to_le_bytes()
                && bytes[16..24] == [0; 8]
                && bytes[24..32] == [1, 0, 1, 0, 6, 0, 0, 0]
                && bytes[32..40] == ((bytes.len() - HEADER - 32) as u64).to_le_bytes()
                && bytes[bytes.len() - 32..] == checksum(&bytes[..bytes.len() - 32]),
            "noncanonical native policy profile",
        )?;
        let roster = &bytes[HEADER..bytes.len() - 32];
        require(
            bytes[40..72] == Sha256::digest(roster)[..],
            "native policy roster hash",
        )?;
        let roster = read_roster(roster, b)?;
        require(
            roster.root_count() == 1,
            "native fill policy requires one root",
        )?;
        Ok(View { roster })
    })
}
fn read_roster<'a>(bytes: &'a [u8], budget: &mut Budget<'_>) -> io::Result<Roster<'a>> {
    read_native_conditional_policy_roster_v1(bytes, MAX_NATIVE_CONDITIONAL_STORAGE_V1, |work| {
        budget.charge_work(work)
    })
    .map_err(|_| io::Error::other("canonical native policy roster refused"))
}
fn checksum(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    hash.finalize().into()
}

/// Reconstructs native content using the caller's independently accepted policy
/// file bytes. This does not establish installation, source acceptance, currentness
/// or protected execution custody: callers must retain those separate original
/// owners. In particular, passing a file copied from the handoff does not make its
/// policy independently accepted. The source and embedded roster must match this
/// exact file before genuine V5 recovery is invoked.
///
/// Policy bytes and handoff backing/metadata are already prepaid on `budget`.
/// Returns the native recovery's full unreserved storage receipt. Failure is
/// terminal and preserves partial reservations; no refund scope encloses recovery.
pub fn recover_native_conditional_handoff_under_policy_file_v1(
    policy_bytes: &[u8],
    handoff: fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5,
    budget: &mut Budget<'_>,
) -> io::Result<(
    crate::RecoveredCompilerConditionalNativeSemanticHandoffV5,
    crate::RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
)> {
    recover_under_policy_file_using(policy_bytes, handoff, None, budget)
}

/// Same exact policy-file/source/roster checks with independently retained CPU
/// expectations. This does not authenticate that roster's enrollment provenance.
/// The file's existing gfx942-only profile and bytes are unchanged. In particular,
/// enrollment native_policy_sha256 is not equated with this policy-file hash.
/// Policy, handoff, metadata and expectations must be prepaid together; failures
/// remain terminal, and no blanket refund scope encloses native recovery.
pub fn recover_native_conditional_handoff_under_policy_file_with_cpu_origins_v1(
    policy_bytes: &[u8],
    handoff: fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5,
    expected_cpu: &[crate::NativeConditionalCpuExpectationV1],
    budget: &mut Budget<'_>,
) -> io::Result<(
    crate::RecoveredCompilerConditionalNativeSemanticHandoffV5,
    crate::RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
)> {
    recover_under_policy_file_using(policy_bytes, handoff, Some(expected_cpu), budget)
}

fn require_cpu_origin_backing(
    policy_bytes: usize,
    handoff_backing: usize,
    expected_cpu: &[crate::NativeConditionalCpuExpectationV1],
    budget: &mut Budget<'_>,
) -> io::Result<()> {
    budget.charge_work(5).map_err(other)?;
    let minimum = [
        policy_bytes,
        handoff_backing,
        fe2o3_compiler_ffi::INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5,
        std::mem::size_of_val(expected_cpu),
    ]
    .into_iter()
    .try_fold(0usize, |n, v| n.checked_add(v))
    .ok_or_else(|| other(ResourceError::Arithmetic))?;
    if budget.storage() < minimum {
        return Err(other(ResourceError::Accounting));
    }
    Ok(())
}

fn recover_under_policy_file_using(
    policy_bytes: &[u8],
    handoff: fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5,
    expected_cpu: Option<&[crate::NativeConditionalCpuExpectationV1]>,
    budget: &mut Budget<'_>,
) -> io::Result<(
    crate::RecoveredCompilerConditionalNativeSemanticHandoffV5,
    crate::RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
)> {
    recover_under_policy_file_selected(
        policy_bytes,
        handoff,
        PolicyCpuSelection::Legacy(expected_cpu),
        budget,
    )
}

#[derive(Clone, Copy)]
enum PolicyCpuSelection<'a> {
    Legacy(Option<&'a [crate::NativeConditionalCpuExpectationV1]>),
    Mapping(&'a crate::NativeConditionalCpuMappingExpectationV1),
}

fn recover_under_policy_file_selected(
    policy_bytes: &[u8],
    handoff: fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5,
    expected_cpu: PolicyCpuSelection<'_>,
    budget: &mut Budget<'_>,
) -> io::Result<(
    crate::RecoveredCompilerConditionalNativeSemanticHandoffV5,
    crate::RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
)> {
    budget.check_prior_denials_v1().map_err(other)?;
    if let PolicyCpuSelection::Legacy(Some(expected)) = expected_cpu {
        require_cpu_origin_backing(
            policy_bytes.len(),
            handoff.backing_capacity(),
            expected,
            budget,
        )?;
    }
    if let PolicyCpuSelection::Mapping(_) = expected_cpu {
        budget.charge_work(5).map_err(other)?;
        let minimum = [
            policy_bytes.len(),
            handoff.backing_capacity(),
            fe2o3_compiler_ffi::INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5,
            size_of::<crate::NativeConditionalCpuMappingExpectationV1>(),
        ]
        .into_iter()
        .try_fold(0usize, |n, v| n.checked_add(v))
        .ok_or_else(|| other(ResourceError::Arithmetic))?;
        if budget.storage() < minimum {
            return Err(other(ResourceError::Accounting));
        }
    }
    let selection_storage = match expected_cpu {
        PolicyCpuSelection::Legacy(Some(_)) => {
            size_of::<Option<&[crate::NativeConditionalCpuExpectationV1]>>() + size_of::<usize>()
        }
        PolicyCpuSelection::Legacy(None) => 0,
        PolicyCpuSelection::Mapping(_) => size_of::<PolicyCpuSelection<'_>>() + size_of::<usize>(),
    };
    if selection_storage != 0 {
        budget.reserve_storage(selection_storage).map_err(other)?;
    }
    budget
        .reserve_storage(size_of::<View<'_>>())
        .map_err(other)?;
    let view = read(policy_bytes, budget)?;
    let source = handoff.capsule().source_packet_bytes();
    budget
        .charge_work(source.len() + view.roster.canonical_bytes().len())
        .map_err(other)?;
    require(
        source.len() as u64 == view.roster.source_packet_len()
            && &<[u8; 32]>::from(Sha256::digest(source)) == view.roster.source_packet_sha256()
            && handoff.capsule().policy_roster_bytes() == view.roster.canonical_bytes(),
        "native source does not match independently accepted policy",
    )?;
    let (roster, storage) =
        crate::reconstruct_inert_native_conditional_policy_roster_in_original_account_v1(
            view.roster.canonical_bytes(),
            source,
            budget,
        )
        .map_err(other)?;
    budget
        .reserve_storage(storage.retained_storage())
        .map_err(other)?;
    budget
        .release_storage(size_of::<View<'_>>())
        .map_err(other)?;
    let result = roster
        .with_root_policies(budget, |roots, b| {
            let limits = fe2o3_kernel_opt::CanonicalRefinedForwardingHistoryLimitsV1::production_v1();
            let profile = fe2o3_amd_target::ProductionAmdTargetProfileV1::Gfx942;
            match expected_cpu {
                PolicyCpuSelection::Legacy(Some(expected)) => crate::recover_compiler_conditional_native_semantic_handoff_with_cpu_origins_in_original_account_v5(
                    handoff, roots, expected, limits, profile, b,
                ),
                PolicyCpuSelection::Legacy(None) => crate::recover_compiler_conditional_native_semantic_handoff_in_original_account_v5(
                    handoff, roots, limits, profile, b,
                ),
                PolicyCpuSelection::Mapping(expected) => crate::recover_compiler_conditional_native_semantic_handoff_with_cpu_mapping_in_original_account_v5(
                    handoff, roots, expected, limits, profile, b,
                ),
            }
            .map_err(|_| io::Error::other("independent native V5 recovery refused"))
        })
        .map_err(other)??;
    drop(roster);
    budget
        .release_storage(
            storage
                .retained_storage()
                .checked_add(selection_storage)
                .ok_or_else(|| other(ResourceError::Arithmetic))?,
        )
        .map_err(other)?;
    Ok(result)
}

fn other(error: impl std::error::Error + Send + Sync + 'static) -> io::Error {
    io::Error::other(error)
}

/// Keeps the exact installed-policy content checks and fixed gfx942 profile,
/// then decodes mapping only from the handoff's actual retained inventory.
/// Independent invocation/policy coordinates remain external acceptance duties;
/// the native-policy identity is not this semantic policy file's hash. Input
/// backing and expectation are prepaid; errors and unwinds remain terminal.
pub fn recover_native_conditional_handoff_under_policy_file_with_cpu_mapping_v1(
    policy_bytes: &[u8],
    handoff: fe2o3_compiler_ffi::InertSemanticCompilerModuleHandoffV5,
    expected: &crate::NativeConditionalCpuMappingExpectationV1,
    budget: &mut Budget<'_>,
) -> io::Result<(
    crate::RecoveredCompilerConditionalNativeSemanticHandoffV5,
    crate::RecoveredCompilerConditionalNativeSemanticHandoffStorageV5,
)> {
    recover_under_policy_file_selected(
        policy_bytes,
        handoff,
        PolicyCpuSelection::Mapping(expected),
        budget,
    )
}
fn require(condition: bool, reason: &'static str) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(io::Error::other(reason))
    }
}
enum ScopeError {
    Resource(ResourceError),
    Io(io::Error),
}
impl From<ResourceError> for ScopeError {
    fn from(value: ResourceError) -> Self {
        Self::Resource(value)
    }
}
fn refundable_scope<T>(
    budget: &mut Budget<'_>,
    floor: usize,
    work: usize,
    scratch: usize,
    run: impl FnOnce(&mut Budget<'_>) -> io::Result<T>,
) -> io::Result<T> {
    budget
        .with_prepaid_scope(floor, 0, work, scratch, |b| run(b).map_err(ScopeError::Io))
        .map_err(|error| match error {
            ScopeError::Resource(error) => other(error),
            ScopeError::Io(error) => error,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe2o3_compiler_lineage::{
        NativeConditionalPolicyRootInputV1 as Root, NativeConditionalPolicyRosterInputV1 as Input,
        encode_native_conditional_policy_roster_v1,
    };
    use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;

    #[test]
    fn origin_selected_policy_input_backing_is_additive_exact_and_one_short() {
        let expected = [crate::NativeConditionalCpuExpectationV1 {
            semantic_root: 9,
            origin: crate::NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
        }];
        let old_inputs = 17
            + 127
            + fe2o3_compiler_ffi::INERT_SEMANTIC_COMPILER_MODULE_HANDOFF_DECODE_METADATA_STORAGE_V5;
        let required = old_inputs + std::mem::size_of_val(&expected);
        for paid in [old_inputs, required - 1, required] {
            let mut work = Work::new(5);
            let mut b = Budget::new(&mut work, required);
            b.reserve_storage(paid).unwrap();
            let ledger = b.work_ledger_identity_v1();
            let result = require_cpu_origin_backing(17, 127, &expected, &mut b);
            assert_eq!(result.is_ok(), paid == required);
            if let Err(error) = result {
                assert!(matches!(
                    error
                        .get_ref()
                        .and_then(|e| e.downcast_ref::<ResourceError>()),
                    Some(ResourceError::Accounting)
                ));
            }
            assert_eq!(b.storage(), paid);
            assert_eq!(b.work(), 5);
            assert!(b.work_ledger_identity_v1() == ledger);
        }
    }

    #[test]
    fn origin_selected_policy_backing_denies_work_and_arithmetic_without_refund() {
        let expected = [crate::NativeConditionalCpuExpectationV1 {
            semantic_root: 9,
            origin: crate::NativeConditionalCpuOriginExpectationV1::SourceRegistrationV1,
        }];
        for work_limit in [4, 5] {
            let mut work = Work::new(work_limit);
            let mut b = Budget::new(&mut work, 19);
            b.reserve_storage(19).unwrap();
            let error = require_cpu_origin_backing(usize::MAX, 1, &expected, &mut b).unwrap_err();
            let resource = error
                .get_ref()
                .and_then(|e| e.downcast_ref::<ResourceError>())
                .unwrap();
            if work_limit == 4 {
                assert!(matches!(resource, ResourceError::Work(_)));
            } else {
                assert!(matches!(resource, ResourceError::Arithmetic));
            }
            assert_eq!(b.storage(), 19);
            assert_eq!(b.work(), if work_limit == 4 { 0 } else { 5 });
            assert_eq!(b.failed_work().is_some(), work_limit == 4);
        }
    }

    #[test]
    fn native_policy_reader_rejects_recorded_denials_before_valid_or_invalid_content() {
        let roster = roster(1);
        let mut work = Work::new(100_000_000);
        let mut b = Budget::new(&mut work, 1_000_000);
        b.reserve_storage(roster.len()).unwrap();
        let (bytes, _) = encode_native_conditional_root_policy_file_v1(&roster, &mut b).unwrap();
        for content in [bytes.as_slice(), &[]] {
            for mode in 0..4 {
                let mut work = Work::new(100_000_000);
                let mut b = Budget::new(&mut work, 1_000_000);
                b.reserve_storage(bytes.capacity()).unwrap();
                if mode == 3 {
                    assert!(b.reserve_storage(1_000_001).is_err());
                }
                if mode != 1 {
                    assert!(b.charge_work(100_000_001).is_err());
                }
                if mode == 1 || mode == 2 {
                    assert!(b.reserve_storage(1_000_001).is_err());
                }
                let before = (
                    b.work(),
                    b.storage(),
                    b.peak_storage(),
                    b.failed_work(),
                    b.failed_storage(),
                );
                let error =
                    validate_native_conditional_root_policy_file_v1(content, &mut b).unwrap_err();
                let resource = error
                    .get_ref()
                    .and_then(|e| e.downcast_ref::<ResourceError>())
                    .unwrap();
                assert!(matches!(
                    (mode, resource),
                    (1, ResourceError::Storage(_)) | (0 | 2 | 3, ResourceError::Work(_))
                ));
                assert_eq!(
                    (
                        b.work(),
                        b.storage(),
                        b.peak_storage(),
                        b.failed_work(),
                        b.failed_storage()
                    ),
                    before
                );
            }
        }
    }
    pub(super) fn roster(count: usize) -> Vec<u8> {
        let signers = [[1; 32]];
        let roots: Vec<_> = (0..count)
            .map(|index| Root {
                semantic_root: index as u32,
                kernel_binding: [2; 32],
                effect_signers: &signers,
                effect_toolchain: [[3; 32]; 5],
                formula_verifying_key: [4; 32],
                formula_toolchain: [[5; 32]; 5],
                formula_boundary: 2,
            })
            .collect();
        encode_native_conditional_policy_roster_v1(
            Input {
                source_packet: b"independently selected source",
                roots: &roots,
            },
            MAX_NATIVE_CONDITIONAL_STORAGE_V1,
            |_| Ok::<_, std::convert::Infallible>(()),
        )
        .unwrap()
    }
    #[test]
    fn native_policy_file_reuses_exact_roster_and_closed_profile() {
        let roster = roster(1);
        let mut work = Work::new(100_000_000);
        let mut b = Budget::new(&mut work, 1_000_000);
        b.reserve_storage(roster.len()).unwrap();
        let (bytes, storage) =
            encode_native_conditional_root_policy_file_v1(&roster, &mut b).unwrap();
        b.reserve_storage(storage).unwrap();
        assert_eq!(bytes.len(), MIN_BYTES);
        let view = read(&bytes, &mut b).unwrap();
        assert_eq!(view.roster.canonical_bytes(), roster);
        assert_eq!(view.roster.root_count(), 1);
        assert_eq!(
            view.roster.source_packet_sha256(),
            &<[u8; 32]>::from(Sha256::digest(b"independently selected source"))
        );
        assert!(!view.roster.grants_authority());
        let floor = b.storage();
        for offset in 0..bytes.len() {
            let mut changed = bytes.clone();
            changed[offset] ^= 1;
            assert!(read(&changed, &mut b).is_err(), "offset {offset}");
            assert_eq!(b.storage(), floor);
        }
        for offset in [8, 10, 16, 24, 26, 28, 30, 32, 40] {
            let mut changed = bytes.clone();
            changed[offset] ^= 1;
            let end = changed.len() - 32;
            let digest = checksum(&changed[..end]);
            changed[end..].copy_from_slice(&digest);
            assert!(read(&changed, &mut b).is_err(), "resealed {offset}");
        }
        assert!(read(&bytes[..bytes.len() - 1], &mut b).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(read(&trailing, &mut b).is_err());
        let mut legacy = bytes;
        legacy[..8].copy_from_slice(b"F3APMS1\0");
        assert!(read(&legacy, &mut b).is_err());
    }
    #[test]
    fn native_policy_file_rejects_multiroot_and_original_budget_denials() {
        let roster = roster(2);
        let mut work = Work::new(100_000_000);
        let mut b = Budget::new(&mut work, 1_000_000);
        b.reserve_storage(roster.len()).unwrap();
        assert!(encode_native_conditional_root_policy_file_v1(&roster, &mut b).is_err());
        let roster = self::roster(1);
        let (bytes, _) = encode_native_conditional_root_policy_file_v1(&roster, &mut b).unwrap();
        let floor = bytes.len();
        for work_limit in [0, 7, 8] {
            let mut work = Work::new(work_limit);
            let mut b = Budget::new(&mut work, 1_000_000);
            b.reserve_storage(floor).unwrap();
            assert!(read(&bytes, &mut b).is_err());
            assert!(b.failed_work().is_some());
            assert_eq!(b.storage(), floor);
        }
        let mut work = Work::new(100_000_000);
        let mut b = Budget::new(&mut work, floor + SCRATCH - 1);
        b.reserve_storage(floor).unwrap();
        assert!(read(&bytes, &mut b).is_err());
        assert!(b.failed_storage().is_some());
        assert_eq!(b.storage(), floor);
    }
}
