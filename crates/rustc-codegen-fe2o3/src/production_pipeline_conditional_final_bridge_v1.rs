//! Same packet visit through actual F/V5 content; original custody stays outside.
#![allow(
    clippy::result_large_err,
    clippy::large_enum_variant,
    reason = "typed terminal causes without a hidden allocation"
)]
use super::{Budget, FinalChain, Owner, Prefix, Profile, Resource};
use crate::compiler_descriptor::conditional_native_v5::{
    ConditionalNativeDescriptorErrorV5, encode_conditional_native_descriptor_v5,
};
use crate::kernel_ir_codegen::{
    InertCompilerModuleTextV1,
    conditional_v5::{ConditionalModuleErrorV5, retain_conditional_compiler_module_text_v5},
};
use crate::production_native_source_lineage_v1::{
    ConditionalPacketErrorV2, PreparedConditionalSourcePacketV2,
    prepare_retained_native_conditional_source_packet_using_v2,
};
use crate::production_pipeline::AuthenticatedProductionBindings;
use crate::production_ranked_projection_v1::{
    ProductionRankedRootProgramV1 as Root, ProductionRankedSemanticProgramV1 as Ranked,
};
use fe2o3_compiler_ffi::{
    COMPILER_DESCRIPTOR_SOURCE_HEADER_STORAGE_V5, COMPILER_DESCRIPTOR_SOURCE_TABLE_STORAGE_V5,
    COMPILER_DESCRIPTOR_SOURCE_VALIDATION_STORAGE_V5, CompilerDescriptorSourceErrorV5,
    CompilerDescriptorSourceV5, compiler_descriptor_source_validation_storage_v5,
};
use fe2o3_compiler_lineage::{ProductionTargetLineageErrorV3, TargetLineageIdentityV3};
use fe2o3_kernel_descriptor::{
    CONDITIONAL_INVOCATION_CODEC_STORAGE_V2, ConditionalInvocationContractV2,
    ConditionalInvocationWireErrorV1, decode_conditional_invocation_contract_v2,
};
use fe2o3_kernel_ir::{
    InertCanonicalKernelIrContractCatalogV1 as Catalog, KernelIrContractCatalogErrorV1,
};
use fe2o3_kernel_opt::{
    InertRefinedForwardingHistoryBytesV1 as Wire, RefinedForwardingHistoryWireErrorV1,
    encode_refined_forwarding_history_v1, materialize_refined_forwarding_history_v1,
    read_refined_forwarding_history_v1,
};
use fe2o3_verifier::{
    NativeConditionalFinalErrorV2, NativeConditionalFinalInputsV2,
    validate_native_conditional_source_through_f_with_cpu_origins_v2,
};
use std::{convert::Infallible, fmt, mem::size_of};

#[path = "production_pipeline_conditional_cpu_origins_v1.rs"]
mod cpu_origins;
#[path = "production_pipeline_conditional_native_handoff_v5.rs"]
pub(super) mod native;
#[path = "production_pipeline_conditional_policy_capture_v1.rs"]
mod policy_capture;

/// Typed but terminal: no source chain for ordinary refund classifiers.
#[derive(Debug)]
pub(crate) enum Error {
    Resource(Resource),
    Packet(ConditionalPacketErrorV2),
    Context(crate::collector::ContextRootVisitErrorV29<Infallible>),
    Contract(ConditionalInvocationWireErrorV1<Resource>),
    History(RefinedForwardingHistoryWireErrorV1),
    Catalog(KernelIrContractCatalogErrorV1),
    Descriptor(ConditionalNativeDescriptorErrorV5),
    Source(CompilerDescriptorSourceErrorV5<Resource>),
    Lowering(dialect_amdgcn::LoweringErrors),
    Layout(dialect_amdgcn::ProductionLlvmLayoutBindingErrorV1),
    Text(ConditionalModuleErrorV5),
    Composition(NativeConditionalFinalErrorV2),
    Lineage(ProductionTargetLineageErrorV3),
    PolicyRoster(fe2o3_compiler_lineage::NativeConditionalPolicyRosterErrorV1<Resource>),
    Native(native::Error),
    Mismatch(&'static str),
}
impl From<Resource> for Error {
    fn from(value: Resource) -> Self {
        Self::Resource(value)
    }
}
impl From<ConditionalPacketErrorV2> for Error {
    fn from(value: ConditionalPacketErrorV2) -> Self {
        Self::Packet(value)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (stage, cause): (&str, &dyn fmt::Display) = match self {
            Self::Resource(e) => ("resource", e),
            Self::Packet(e) => ("packet", e),
            Self::Context(e) => return write!(f, "conditional F/V5 producer context: {e:?}"),
            Self::Contract(e) => ("contract", e),
            Self::History(e) => ("history", e),
            Self::Catalog(e) => ("catalog", e),
            Self::Descriptor(e) => ("descriptor", e),
            Self::Source(e) => ("descriptor source", e),
            Self::Lowering(e) => ("LLVM lowering", e),
            Self::Layout(e) => ("LLVM layout", e),
            Self::Text(e) => ("text", e),
            Self::Composition(e) => ("composition", e),
            Self::Lineage(e) => ("lowering identity", e),
            Self::PolicyRoster(e) => {
                return write!(f, "conditional F/V5 producer policy roster: {e:?}");
            }
            Self::Native(e) => ("native handoff", e),
            Self::Mismatch(e) => ("subjects", e),
        };
        write!(f, "conditional F/V5 producer {stage}: {cause}")
    }
}
impl std::error::Error for Error {}

/// Retained beside the original chain/collector, never a native handoff owner.
pub(super) struct RetainedFinalContentV5 {
    history: Wire,
    catalog: Catalog,
    descriptor: CompilerDescriptorSourceV5,
    module: InertCompilerModuleTextV1,
    pre_descriptor_llvm: TargetLineageIdentityV3,
    policy_roster: Vec<u8>,
    cpu_origins: cpu_origins::RetainedCpuOriginRosterV1,
    retained: usize,
}

const IDENTITY_SCRATCH: usize = size_of::<sha2::Sha256>()
    + 32
    + size_of::<Result<TargetLineageIdentityV3, ProductionTargetLineageErrorV3>>();

fn raw_identity(bytes: &[u8], budget: &mut Budget<'_>) -> Result<TargetLineageIdentityV3, Error> {
    use sha2::{Digest, Sha256};
    budget.reserve_storage(IDENTITY_SCRATCH)?;
    budget.charge_work(bytes.len().checked_add(128).ok_or(Resource::Arithmetic)?)?;
    let length = u64::try_from(bytes.len()).map_err(|_| Resource::Arithmetic)?;
    let identity = TargetLineageIdentityV3::new(Sha256::digest(bytes).into(), length)
        .map_err(Error::Lineage)?;
    budget.release_storage(IDENTITY_SCRATCH)?;
    Ok(identity)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ObservedContentV5([(usize, [u8; 32]); 5]);

#[cfg(test)]
impl ObservedContentV5 {
    // Fixed-size diagnostic stamps only, not proof or a new storage receipt.
    pub(super) fn from_bytes(bytes: [&[u8]; 5]) -> Self {
        use sha2::{Digest, Sha256};
        Self(bytes.map(|bytes| (bytes.len(), Sha256::digest(bytes).into())))
    }
}

#[cfg(test)]
impl RetainedFinalContentV5 {
    fn observation(&self, packet: &[u8]) -> ObservedContentV5 {
        ObservedContentV5::from_bytes([
            packet,
            self.history.canonical_bytes(),
            self.catalog.canonical_bytes(),
            self.descriptor.canonical_bytes(),
            self.module.llvm_ir().as_bytes(),
        ])
    }

    pub(super) fn assert_observed_v5(
        &self,
        value: &super::ConditionalPrefixForFV1,
        called: ObservedContentV5,
    ) {
        assert_eq!(self.observation(value.packet.source_packet()), called);
        assert!(!self.history.canonical_bytes().is_empty());
        assert!(!self.catalog.canonical_bytes().is_empty());
        assert!(!self.descriptor.canonical_bytes().is_empty());
        assert!(!self.descriptor.authenticates_compiler_origin());
        assert!(!self.descriptor.grants_launch_authority());
        assert!(self.module.llvm_ir().contains(".fe2o3.kd.v5"));
        assert!(!self.module.llvm_ir().contains(".fe2o3.kd.v3"));
        assert!(!self.module.llvm_ir().contains(".fe2o3.kd.v4"));
        assert_eq!(
            self.module.descriptor_binding_version_for_test_v3(),
            Some(5)
        );

        let ranked = &value.preparation.ranked;
        let roster = fe2o3_compiler_lineage::read_native_conditional_policy_roster_v1(
            &self.policy_roster,
            fe2o3_compiler_lineage::MAX_NATIVE_CONDITIONAL_STORAGE_V1,
            |_| Ok::<_, Infallible>(()),
        )
        .unwrap();
        use sha2::{Digest, Sha256};
        assert_eq!(
            roster.source_packet_sha256(),
            &<[u8; 32]>::from(Sha256::digest(value.packet.source_packet())),
        );
        assert_eq!(
            roster.source_packet_len(),
            value.packet.source_packet().len() as u64
        );
        assert_eq!(roster.root_count(), ranked.roots().len());
        for (row, root) in roster.roots().zip(ranked.roots()) {
            let producer = root.conditional_producer_inputs_v2().unwrap();
            assert_eq!(row.semantic_root(), root.semantic_root().index());
            assert_eq!(row.kernel_binding(), root.kernel_binding());
            assert_eq!(
                row.effect_signers(),
                producer
                    .effect_policy
                    .signer_identities()
                    .iter()
                    .map(|signer| *signer.as_bytes())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                row.effect_toolchain(),
                &policy_capture::toolchain(producer.effect_policy.toolchain())
            );
            assert_eq!(
                row.formula_verifying_key(),
                producer.formula_policy.verifying_key()
            );
            assert_eq!(
                row.formula_toolchain(),
                &policy_capture::toolchain(producer.formula_policy.toolchain())
            );
            assert_eq!(
                row.formula_boundary(),
                producer.formula_policy.boundary() as u8
            );
        }
        let original = ranked.materialized();
        let proof = value.packet.proof();
        let replayed = proof.source().source();
        let semantic = original.semantic_ssa().source_semantic();
        assert_eq!(
            replayed
                .semantic_ssa()
                .source_semantic()
                .canonical_encoding(),
            semantic.canonical_encoding(),
        );
        assert_eq!(
            replayed.executable().canonical().canonical_bytes(),
            original.executable().canonical().canonical_bytes(),
        );
        assert_eq!(replayed.source_launch(), original.source_launch());
        assert_eq!(
            self.catalog.semantic_source(),
            semantic.semantic_sha256().as_bytes()
        );
        assert!(self.catalog.definitions().is_empty());
        assert!(self.catalog.bindings().is_empty());

        // Passive bounded queries, outside the production resource measurement.
        // No budget, importer, semantic replay or authenticated owner is created.
        let mut charge = |_| Ok::<(), Resource>(());
        let table = self
            .descriptor
            .table(
                self.descriptor
                    .storage()
                    .retained_storage()
                    .checked_add(COMPILER_DESCRIPTOR_SOURCE_TABLE_STORAGE_V5)
                    .unwrap(),
                &mut charge,
            )
            .unwrap();
        let roots = &value.preparation.bindings.typed_descriptor_roots;
        assert_eq!(table.kernel_count(), roots.len());
        assert_eq!(proof.root_count(), roots.len());
        assert_eq!(proof.canonical_kernel_order().len(), roots.len());
        assert_eq!(self.module.kernel_entries().len(), roots.len());
        for (position, &ordinal) in proof.canonical_kernel_order().iter().enumerate() {
            let ordinal = ordinal as usize;
            let root = &roots[ordinal];
            let row = table.kernel(position, &mut charge).unwrap();
            assert_eq!(*row.kernel_id().as_bytes(), root.kernel_binding_bytes());
            assert_eq!(row.entry_name(), root.entry_symbol());
            assert!(
                self.module
                    .kernel_entries()
                    .iter()
                    .any(|entry| entry == row.entry_name())
            );
            let contract = row.conditional_contract(&mut charge).unwrap();
            let producer = ranked.roots()[ordinal]
                .conditional_producer_inputs_v2()
                .unwrap();
            assert_eq!(
                contract.canonical_bytes(),
                producer.contract.canonical_bytes()
            );
            assert_eq!(contract.subjects().kernel_id, root.kernel_binding_bytes());
            assert_eq!(
                contract.subjects().source_semantic_identity,
                *semantic.semantic_sha256().as_bytes()
            );
            let report = proof.formula_report(ordinal).unwrap();
            let theorem = contract.theorem();
            assert_eq!(
                &theorem.statement_identity,
                report.statement_identity().as_bytes()
            );
            assert_eq!(
                &theorem.generated_source_identity,
                report.generated_source_identity().as_bytes()
            );
            assert_eq!(
                &theorem.execution_identity,
                report.execution_identity().as_bytes()
            );
            assert_eq!(
                &theorem.receipt_identity,
                report.receipt_identity().as_bytes()
            );
            assert_eq!(
                &theorem.cpu_input_commitment,
                report.cpu_input_commitment().as_bytes()
            );
        }
    }
}

#[cfg(test)]
fn observe_history(
    frame: &fe2o3_kernel_opt::InertRefinedForwardingHistoryRefV1<'_>,
    chain: &FinalChain,
    bound: &Owner,
    checked: &Prefix,
) {
    use fe2o3_kernel_opt::RefinedForwardingHistoryRoleV1 as Role;
    let inputs = chain.inputs(bound, checked);
    let p8 = inputs.prefix;
    let p7 = p8.prefix;
    let p6 = p7.prefix;
    let p5 = p6.prefix;
    for (role, owner) in [
        (Role::B, p5.input),
        (Role::C, p5.intermediate),
        (Role::S, p5.stored),
        (Role::O, p5.output),
        (Role::I, p6.output),
        (Role::J, p7.output),
        (Role::K, p8.output),
        (Role::P, inputs.promoted),
        (Role::H, inputs.preheaders),
        (Role::L, inputs.licm),
        (Role::R, inputs.refined),
        (Role::F, inputs.output),
    ] {
        assert_eq!(
            frame.graph_bytes(role),
            owner.canonical().canonical_bytes(),
            "{role:?}"
        );
    }
    assert_eq!(frame.limits(), inputs.limits);
    assert_eq!(frame.limits(), chain.expected_limits());
}

pub(super) fn prepare(
    ranked: Ranked,
    bindings: &AuthenticatedProductionBindings,
    bound: &Owner,
    checked: &Prefix,
    chain: &FinalChain,
    budget: &mut Budget<'_>,
) -> Result<
    (
        Ranked,
        PreparedConditionalSourcePacketV2,
        RetainedFinalContentV5,
    ),
    Error,
> {
    let result = prepare_retained_native_conditional_source_packet_using_v2(
        ranked,
        &bindings.typed_descriptor_roots,
        budget,
        |source, roots, order, packet, policies, budget| {
            // These are the original producer owners, not identities recovered
            // from the packet. Content agreement cannot replace their custody.
            chain.check_owned(budget)?;
            let semantic = source.semantic_ssa().source_semantic();
            if bindings
                .context_entries
                .materialization_source_v29(semantic, budget)
                .map_err(Error::Context)?
                .is_some()
            {
                return Err(Error::Mismatch(
                    "conditional F context argument elision unsupported",
                ));
            }
            budget.reserve_storage(
                header()?
                    .checked_add(size_of::<Error>())
                    .ok_or(Resource::Arithmetic)?,
            )?;
            let cpu_origins = cpu_origins::capture(bindings, roots, budget)?;
            let policy_roster = policy_capture::capture(
                packet,
                policies,
                roots
                    .iter()
                    .map(|root| (root.semantic_root().index(), *root.kernel_binding())),
                budget,
            )?;
            let history =
                encode_refined_forwarding_history_v1(chain.inputs(bound, checked), budget)
                    .map_err(Error::History)?;
            budget.reserve_storage(history.storage().retained_storage())?;
            let (catalog, storage) = Catalog::from_rows_with_budget(
                *semantic.semantic_sha256().as_bytes(),
                &[],
                &[],
                budget,
            )
            .map_err(Error::Catalog)?;
            budget.reserve_storage(storage.retained_storage())?;
            // As on the existing Direct path, the empty catalog admits no
            // Execution/verification marker. The full relation checks actual F.
            let contracts = contracts(roots, order, budget)?;
            let (bytes, receipt) = encode_conditional_native_descriptor_v5(
                &bindings.typed_descriptor_roots,
                semantic,
                chain.output(),
                &bindings.rustc_target,
                &contracts,
                budget,
            )
            .map_err(Error::Descriptor)?;
            budget.reserve_storage(receipt.retained_storage())?;
            let descriptor = adopt_descriptor(bytes, budget)?;
            let (prefix, prefix_storage) =
                emit_prefix(chain.output(), bindings.rustc_target.profile(), budget)?;
            // Retain the actual producer preimage coordinate before its backing
            // dies; recovery compares it with the same-visit checked relation.
            let pre_descriptor_llvm = raw_identity(prefix.as_bytes(), budget)?;
            let (module, module_storage) = retain_conditional_compiler_module_text_v5(
                chain.output(),
                &prefix,
                &descriptor,
                budget,
            )
            .map_err(Error::Text)?;
            budget.reserve_storage(module_storage.retained_storage())?;
            drop(prefix);
            budget.release_storage(prefix_storage)?;
            drop(contracts);
            let policy_storage = policy_capture::retained_storage(&policy_roster)?;
            let origin_storage = cpu_origins.retained_storage()?;
            let retained = header()?
                .checked_add(history.storage().retained_storage())
                .and_then(|n| n.checked_add(storage.retained_storage()))
                .and_then(|n| n.checked_add(descriptor.storage().retained_storage()))
                .and_then(|n| n.checked_add(module_storage.retained_storage()))
                .and_then(|n| n.checked_add(policy_storage))
                .and_then(|n| n.checked_add(origin_storage))
                .ok_or(Resource::Arithmetic)?;
            let content = RetainedFinalContentV5 {
                history,
                catalog,
                descriptor,
                module,
                pre_descriptor_llvm,
                policy_roster,
                cpu_origins,
                retained,
            };
            let frame =
                read_refined_forwarding_history_v1(content.history.canonical_bytes(), budget)
                    .map_err(Error::History)?;
            budget.reserve_storage(frame.storage().retained_storage())?;
            let decoded = materialize_refined_forwarding_history_v1(&frame, budget)
                .map_err(Error::History)?;
            budget.reserve_storage(decoded.storage().retained_storage())?;
            budget.reserve_storage(COMPILER_DESCRIPTOR_SOURCE_TABLE_STORAGE_V5)?;
            let declared = content
                .descriptor
                .storage()
                .retained_storage()
                .checked_add(COMPILER_DESCRIPTOR_SOURCE_TABLE_STORAGE_V5)
                .ok_or(Resource::Arithmetic)?;
            let descriptors = content
                .descriptor
                .table(declared, &mut |n| budget.charge_work(n))
                .map_err(Error::Source)?;
            // Exactly one independent consumer visit. There is no preceding C1
            // source-only import and no second execution of the runtime formula.
            #[cfg(test)]
            if super::tests::observing_bridge() {
                observe_history(&frame, chain, bound, checked);
                super::tests::bridge_consumer_called(content.observation(packet));
            }
            let (proof, storage) =
                validate_native_conditional_source_through_f_with_cpu_origins_v2(
                    packet,
                    policies,
                    content.cpu_origins.rows(),
                    NativeConditionalFinalInputsV2 {
                        decoded_history: &decoded,
                        expected_limits: chain.expected_limits(),
                        final_catalog: &content.catalog,
                        published_output_bytes: chain.output().canonical().canonical_bytes(),
                        profile: bindings.rustc_target.profile(),
                        descriptors: &descriptors,
                        final_llvm: content.module.llvm_ir(),
                    },
                    budget,
                )
                .map_err(Error::Composition)?;
            // Only destruction follows before the common packet assembler pays
            // the returned unreserved source receipt. Scratch refunds occur only
            // after source and target postchecks and all these borrows are gone.
            drop(descriptors);
            drop(decoded);
            drop(frame);
            let retained = content.retained;
            Ok((proof, storage, content, retained))
        },
    );
    #[cfg(test)]
    if result.is_ok() {
        // Includes the original source/target postchecks, before installation.
        super::tests::bridge_completed();
    }
    result
}

fn header() -> Result<usize, Resource> {
    size_of::<RetainedFinalContentV5>()
        .checked_sub(size_of::<Wire>())
        .and_then(|n| n.checked_sub(size_of::<Catalog>()))
        .and_then(|n| n.checked_sub(size_of::<CompilerDescriptorSourceV5>()))
        .and_then(|n| n.checked_sub(size_of::<InertCompilerModuleTextV1>()))
        .and_then(|n| n.checked_sub(size_of::<Vec<u8>>()))
        .and_then(|n| n.checked_sub(size_of::<cpu_origins::RetainedCpuOriginRosterV1>()))
        .ok_or(Resource::Arithmetic)
}

fn contracts<'a>(
    roots: &'a [Root],
    order: &[u32],
    budget: &mut Budget<'_>,
) -> Result<Vec<ConditionalInvocationContractV2<'a>>, Error> {
    budget.charge_work(1)?;
    if roots.len() != order.len() {
        return Err(Error::Mismatch("complete canonical contract roster"));
    }
    let bytes = roots
        .len()
        .checked_mul(size_of::<ConditionalInvocationContractV2<'_>>())
        .and_then(|n| n.checked_add(size_of::<Vec<ConditionalInvocationContractV2<'_>>>()))
        .and_then(|n| n.checked_add(CONDITIONAL_INVOCATION_CODEC_STORAGE_V2))
        .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(bytes)?;
    let mut contracts = Vec::new();
    contracts
        .try_reserve_exact(roots.len())
        .map_err(|_| Resource::Allocation)?;
    if contracts.capacity() != roots.len() {
        return Err(Resource::Accounting.into());
    }
    for index in order {
        budget.charge_work(1)?;
        let root = roots
            .get(*index as usize)
            .ok_or(Error::Mismatch("canonical root ordinal"))?;
        let input = root
            .conditional_producer_inputs_v2()
            .ok_or(Error::Mismatch("retained conditional contract"))?;
        contracts.push(
            decode_conditional_invocation_contract_v2(input.contract.canonical_bytes(), &mut |n| {
                budget.charge_work(n)
            })
            .map_err(Error::Contract)?,
        );
    }
    Ok(contracts)
}

fn adopt_descriptor(
    bytes: Vec<u8>,
    budget: &mut Budget<'_>,
) -> Result<CompilerDescriptorSourceV5, Error> {
    // Transfer the already-paid Vec header/capacity, adding only the owner
    // fields and simultaneous validation scratch, never copying the bytes.
    let extra = COMPILER_DESCRIPTOR_SOURCE_HEADER_STORAGE_V5
        .checked_sub(size_of::<Vec<u8>>())
        .and_then(|n| n.checked_add(COMPILER_DESCRIPTOR_SOURCE_VALIDATION_STORAGE_V5))
        .ok_or(Resource::Arithmetic)?;
    let declared = compiler_descriptor_source_validation_storage_v5(bytes.capacity())
        .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(extra)?;
    CompilerDescriptorSourceV5::from_owned_canonical_bytes(bytes, declared, &mut |n| {
        budget.charge_work(n)
    })
    .map_err(Error::Source)
}

fn emit_prefix(
    output: &Owner,
    profile: Profile,
    budget: &mut Budget<'_>,
) -> Result<(String, usize), Error> {
    // The existing emitter has its own resource policy. Prepay both simultaneous
    // bounded String owners here; the adapter and replay pay their own scratch.
    let native_limit = dialect_amdgcn::MAX_COMPILER_MODULE_TEXT_BYTES;
    let bound_limit = dialect_amdgcn::MAX_PRODUCTION_SEMANTIC_ANCHOR_LLVM_TEXT_BYTES_V1;
    let prepaid = native_limit
        .checked_add(bound_limit)
        .and_then(|n| n.checked_add(2 * size_of::<String>()))
        .ok_or(Resource::Arithmetic)?;
    budget.reserve_storage(prepaid)?;
    budget.charge_work(output.canonical().canonical_bytes().len())?;
    let native = match profile {
        Profile::Gfx942 => dialect_amdgcn::lower_canonical_v12_compiler_module_to_gfx942_xnack_minus_llvm_ir_with_semantic_anchors_v1(output),
        Profile::Gfx950 => dialect_amdgcn::lower_canonical_v12_compiler_module_to_gfx950_xnack_minus_llvm_ir_with_semantic_anchors_v1(output),
    }.map_err(Error::Lowering)?;
    let native_storage = reconcile_text(&native, native_limit, budget)?;
    budget.charge_work(native.len())?;
    let bound =
        dialect_amdgcn::bind_production_llvm22_worker_layout_v1(&native).map_err(Error::Layout)?;
    let bound_storage = reconcile_text(&bound, bound_limit, budget)?;
    drop(native);
    budget.release_storage(native_storage)?;
    Ok((bound, bound_storage))
}

// As in the existing text replay, reconcile the real allocator capacity, not
// just length; only unused reservation or already-destroyed owners are released.
fn reconcile_text(text: &String, maximum: usize, budget: &mut Budget<'_>) -> Result<usize, Error> {
    let actual = size_of::<String>()
        .checked_add(text.capacity())
        .ok_or(Resource::Arithmetic)?;
    if text.capacity() > maximum {
        budget.reserve_storage(text.capacity() - maximum)?;
    } else {
        budget.release_storage(maximum - text.capacity())?;
    }
    if text.len() > maximum {
        return Err(Error::Mismatch("bounded canonical LLVM prefix"));
    }
    Ok(actual)
}

#[cfg(test)]
#[path = "production_pipeline_conditional_final_bridge_v1_tests.rs"]
mod tests;
