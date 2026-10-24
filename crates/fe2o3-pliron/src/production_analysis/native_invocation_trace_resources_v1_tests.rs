use super::super::pliron_invocation_trace::native_resources_v1::{awi_backing, fold_scratch};
use super::tests::{noop, with_checked};
use super::*;
use fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as Work;
use std::collections::HashMap;

#[test]
fn vector_component_exact_and_one_short_are_source_derived() {
    // Three u64 elements and one Vec header on the supported 64-bit host.
    const WORK: usize = 1;
    const BYTES: usize = 24 + 3 * 8;
    for (work_limit, storage_limit, accepted) in [
        (WORK, BYTES, true),
        (WORK - 1, BYTES, false),
        (WORK, BYTES - 1, false),
    ] {
        let mut work = Work::new(work_limit);
        let mut budget = Budget::new(&mut work, storage_limit);
        let result = reserve_rows::<u64>(3, &mut budget);
        assert_eq!(result.is_ok(), accepted);
        if accepted {
            assert_eq!(budget.work(), WORK);
            assert_eq!(budget.storage(), BYTES);
            drop(result);
            budget.release_storage(BYTES).unwrap();
        }
    }
}

#[test]
fn apint_component_covers_inline_and_double_width_backing() {
    for (width, bytes) in [(1, 0), (8, 0), (32, 0), (64, 0), (128, 16), (256, 32)] {
        assert_eq!(awi_backing(width).unwrap(), bytes);
    }
    assert!(awi_backing(0).is_err());
    assert!(awi_backing(257).is_err());
    // 128-bit checked multiply: four boxed attrs, two whole attribute clones,
    // six 256-bit temporaries and five 128-bit value clones/truncations/product.
    use pliron::{attribute::AttrObj, builtin::attributes::IntegerAttr, utils::apint::APInt};
    let expected = 2 * size_of::<Vec<Option<AttrObj>>>()
        + 4 * size_of::<Option<AttrObj>>()
        + 6 * (size_of::<IntegerAttr>() + 16)
        + 6 * (size_of::<APInt>() + 32)
        + 5 * (size_of::<APInt>() + 16);
    assert_eq!(fold_scratch(128, 2, 2).unwrap(), expected);
    for cap in [expected, expected - 1] {
        let mut work = Work::new(1);
        let mut budget = Budget::new(&mut work, cap);
        assert_eq!(budget.reserve_storage(expected).is_ok(), cap == expected);
    }
}

#[test]
fn protected_component_drops_payloads_before_refund_and_retains_work() {
    let mut work = Work::new(5);
    let mut budget = Budget::new(&mut work, 17);
    budget.reserve_storage(4).unwrap();
    let result = protected(&mut budget, |budget| {
        budget.reserve_storage(13)?;
        budget.charge_work(3)?;
        Err::<(), _>(Failure::Callback("component"))
    });
    assert!(matches!(result, Err(Failure::Callback("component"))));
    assert_eq!(budget.work(), 5);
    assert_eq!(budget.storage(), 4);
    assert_eq!(budget.peak_storage(), 17);
}

#[test]
fn first_query_error_discards_a_rejected_return_value_before_recovery() {
    struct Dropped<'a>(&'a Cell<bool>);
    impl Drop for Dropped<'_> {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    with_checked(&noop(), |checked, budget| {
        let dropped = Cell::new(false);
        let floor = budget.storage();
        let result = with_canonical_invocation_traces_v1(checked, budget, |view, budget| {
            let _ = view.attempt(4, budget);
            Ok(Dropped(&dropped))
        });
        assert!(matches!(
            result,
            Err(CanonicalInvocationTraceErrorV1 {
                failure: Failure::InvalidQuery { function: 4 }
            })
        ));
        assert!(dropped.get());
        assert_eq!(budget.storage(), floor);
    });
}

#[test]
fn storage_floor_undercut_is_not_refunded_twice() {
    with_checked(&noop(), |checked, budget| {
        let floor = budget.storage();
        let error = with_canonical_invocation_traces_v1::<()>(checked, budget, |_, budget| {
            budget.release_storage(1)?;
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(
            error.failure,
            Failure::Resource(Resource::Accounting)
        ));
        assert_eq!(budget.storage(), floor);
    });
}

#[test]
fn nested_panic_payload_destruction_finishes_before_scope_recovery() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Payload {
        remaining: usize,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if self.remaining != 0 {
                std::panic::panic_any(Payload {
                    remaining: self.remaining - 1,
                    drops: self.drops.clone(),
                });
            }
        }
    }
    with_checked(&noop(), |checked, budget| {
        let drops = Arc::new(AtomicUsize::new(0));
        let floor = budget.storage();
        let error = with_canonical_invocation_traces_v1::<()>(checked, budget, |_, _| {
            std::panic::panic_any(Payload {
                remaining: 2,
                drops: drops.clone(),
            });
        })
        .unwrap_err();
        assert!(matches!(error.failure, Failure::Panicked));
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        assert_eq!(budget.storage(), floor);
        with_canonical_invocation_traces_v1(checked, budget, |_, _| Ok(())).unwrap();
    });
}

#[test]
fn moving_the_same_budget_ledger_to_a_new_slot_refuses_queries() {
    let mut work = Work::new(1);
    let budget = Budget::new(&mut work, 0);
    let guard = Guard::new(&budget);
    let mut moved = Box::new(budget);
    assert!(matches!(
        guard.query(&mut moved),
        Err(Failure::Resource(Resource::Accounting))
    ));
    assert_eq!(moved.work(), 0);
    assert_eq!(moved.storage(), 0);
}

// Four closed, independently specified source profiles. No production census,
// envelope helper, receipt, trace result or observed counter chooses a limit.
#[derive(Clone, Copy, Debug)]
enum Profile {
    Empty,
    Read,
    Mixed,
    Later,
}
impl Profile {
    fn shape(self) -> (usize, usize, usize, usize, usize, usize, usize) {
        // B, nonterminators, block arguments including function parameters,
        // results, operand occurrences, edges, edge payload occurrences.
        match self {
            Self::Empty => (1, 0, 0, 0, 0, 0, 0),
            Self::Read => (1, 1, 1, 1, 1, 0, 0),
            Self::Mixed => (1, 2, 1, 1, 1, 0, 0),
            Self::Later => (2, 3, 2, 2, 3, 1, 1),
        }
    }
    fn events(self) -> usize {
        match self {
            Self::Empty => 1,
            Self::Mixed => 3,
            _ => 2,
        }
    }
    fn source(self) -> fe2o3_kernel_ir::Module {
        use fe2o3_kernel_ir::*;
        let mut module = tests::noop();
        module.id = ModuleId::new("native-resource");
        let function = &mut module.functions[0];
        let body = function.body.as_mut().unwrap();
        body.blocks[0].id = BlockId(0);
        if !matches!(self, Self::Empty) {
            function.signature.parameters = vec![Type::pointer(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Global,
                AccessMode::ReadWrite,
            )];
            body.parameters = vec![ValueId(0)];
        }
        let scalar = Type::Scalar(ScalarType::U32);
        let access = MemoryAccess::new(AddressSpace::Global, 4);
        match self {
            Self::Empty => {}
            Self::Read => body.blocks[0].operations.push(Operation::effect_free(
                ValueDef::new(ValueId(1), scalar),
                OperationKind::Load {
                    pointer: ValueId(0),
                    access,
                },
            )),
            Self::Mixed => {
                module.kernels[0].workgroup_size = Some(WorkgroupSize::new(1, 1, 1));
                body.blocks[0].operations = vec![
                    Operation::new(
                        vec![ValueDef::new(ValueId(1), scalar)],
                        OperationKind::Atomic(Atomic {
                            kind: AtomicKind::Load,
                            pointer: ValueId(0),
                            value: None,
                            compare: None,
                            access,
                            scope: SynchronizationScope::Device,
                            ordering: MemoryOrdering::Acquire,
                            failure_ordering: None,
                        }),
                    ),
                    Operation::new(
                        vec![],
                        OperationKind::Barrier(Barrier {
                            execution_scope: SynchronizationScope::Workgroup,
                            memory_scope: SynchronizationScope::Workgroup,
                            semantics: BarrierSemantics::new(
                                MemoryOrdering::AcquireRelease,
                                [AddressSpace::Workgroup, AddressSpace::Global],
                            ),
                        }),
                    ),
                ];
            }
            Self::Later => {
                body.blocks[0].operations.push(Operation::effect_free(
                    ValueDef::new(ValueId(1), scalar.clone()),
                    OperationKind::Constant(Constant::U32(7)),
                ));
                body.blocks[0].terminator = Some(Terminator::Branch {
                    target: BlockId(1),
                    arguments: vec![ValueId(1)],
                });
                let mut later = BasicBlock::new(BlockId(1));
                later
                    .parameters
                    .push(ValueDef::new(ValueId(2), scalar.clone()));
                later.operations = vec![
                    Operation::effect_free(
                        ValueDef::new(ValueId(3), scalar),
                        OperationKind::Constant(Constant::U32(9)),
                    ),
                    Operation::new(
                        vec![],
                        OperationKind::Store {
                            pointer: ValueId(0),
                            value: ValueId(2),
                            access,
                        },
                    ),
                ];
                later.terminator = Some(Terminator::Return { values: vec![] });
                body.blocks.push(later);
            }
        }
        module
    }
    fn inverse(self) -> (usize, usize, usize, usize) {
        use fe2o3_kernel_ir::{BasicBlock, Function, Kernel, Operation, Type, ValueDef, ValueId};
        let (b, n, ba, r, _, _, payloads) = self.shape();
        // Wire fields are independently listed in RESOURCE_DERIVATION.md.
        // B-tree role scratch is 3*(11*8+14*8)+4*8 = 632 on this host.
        let (bytes, tokens, semantic, semantic_peak) = match self {
            Self::Empty => (127, 33, 205, 21),
            Self::Read => (157, 48, 272, 50),
            Self::Mixed => (189, 65, 299, 50),
            Self::Later => (219, 71, 525, 104),
        };
        assert!(semantic_peak < 632);
        let parameter = usize::from(!matches!(self, Self::Empty));
        let mut payload = 29
            + size_of::<Function>()
            + size_of::<Kernel>()
            + b * size_of::<BasicBlock>()
            + parameter * (2 * size_of::<Type>() + size_of::<ValueId>())
            + (ba - parameter + r) * size_of::<ValueDef>()
            + n * size_of::<Operation>()
            + payloads * size_of::<ValueId>();
        if matches!(self, Self::Mixed) {
            let address = size_of::<fe2o3_kernel_ir::AddressSpace>();
            payload += 2 * (3 * (11 * address + 14 * 8) + 4 * address);
        }
        let retained = size_of::<Owner>() + bytes + payload;
        (bytes, tokens, semantic, retained)
    }
}

#[derive(Clone, Copy, Debug)]
enum Debit {
    Work(usize),
    Reserve(usize),
    Release(usize),
}
#[derive(Clone, Copy, Debug)]
struct Step {
    label: &'static str,
    debit: Debit,
    aggregate: bool,
}
#[derive(Default)]
struct Oracle {
    steps: Vec<Step>,
    live: usize,
    event_capacity: usize,
}
#[derive(Debug)]
struct Prediction {
    work: usize,
    peak: usize,
    retained: usize,
    denial: Option<(&'static str, bool, usize)>,
}
impl Oracle {
    fn work(&mut self, amount: usize, label: &'static str) {
        self.steps.push(Step {
            label,
            debit: Debit::Work(amount),
            aggregate: false,
        });
    }
    fn aggregate_work(&mut self, amount: usize) {
        self.steps.push(Step {
            label: "existing.admitted.component",
            debit: Debit::Work(amount),
            aggregate: true,
        });
    }
    fn reserve(&mut self, amount: usize, label: &'static str) {
        self.steps.push(Step {
            label,
            debit: Debit::Reserve(amount),
            aggregate: false,
        });
        self.live += amount;
    }
    fn release(&mut self, amount: usize) {
        self.steps.push(Step {
            label: "drop.before.refund",
            debit: Debit::Release(amount),
            aggregate: false,
        });
        self.live -= amount;
    }
    fn rows<T>(&mut self, count: usize, label: &'static str) {
        self.work(1, label);
        self.reserve(size_of::<Vec<T>>() + count * size_of::<T>(), label);
        self.reserve(0, "exact.capacity.reconciliation");
    }
    fn map<K, V>(&mut self, count: usize, label: &'static str) -> usize {
        let bytes = size_of::<HashMap<K, V>>() + (2 * count + 1) * (size_of::<(K, V)>() + 16);
        self.work(1, label);
        self.reserve(bytes, label);
        bytes
    }
    fn predict(&self, work_limit: usize, storage_limit: usize) -> Prediction {
        let mut out = Prediction {
            work: 0,
            peak: 0,
            retained: 0,
            denial: None,
        };
        for step in &self.steps {
            match step.debit {
                Debit::Work(amount) => {
                    let next = out.work + amount;
                    if next > work_limit {
                        assert!(
                            !step.aggregate,
                            "an interior component work cut is not derived: {}",
                            step.label
                        );
                        out.denial = Some((step.label, true, next));
                        return out;
                    }
                    out.work = next;
                }
                Debit::Reserve(amount) => {
                    let next = out.retained + amount;
                    if next > storage_limit {
                        out.denial = Some((step.label, false, next));
                        return out;
                    }
                    out.retained = next;
                    out.peak = out.peak.max(next);
                }
                Debit::Release(amount) => out.retained -= amount,
            }
        }
        out
    }
    fn work_cut(&self, label: &str) -> usize {
        let mut work = 0;
        for step in &self.steps {
            if let Debit::Work(amount) = step.debit {
                if step.label == label {
                    assert_ne!(amount, 0);
                    return work + amount - 1;
                }
                work += amount;
            }
        }
        panic!("missing independently selected cut {label}")
    }
    fn schema(&mut self, keys: &[&str]) {
        self.work(1, "schema.header");
        for key in keys {
            self.work(key.len() + keys.len(), "schema.key");
        }
    }
    fn inverse(&mut self, profile: Profile, phase: &'static str) -> usize {
        use fe2o3_kernel_ir::{
            VERIFIED_CANONICAL_KERNEL_IR_V12_IDENTITY_DOMAIN_V1, VerifiedCanonicalKernelIrV12,
        };
        let (bytes, tokens, semantic, retained) = profile.inverse();
        let canonical = size_of::<VerifiedCanonicalKernelIrV12>() + bytes;
        self.aggregate_work(tokens);
        self.reserve(canonical, "inverse.canonical");
        self.reserve(632, "inverse.encoder.roles");
        self.aggregate_work(tokens + 21 + bytes + 4);
        self.release(632);
        self.reserve(retained - canonical, "inverse.decoded.payload");
        self.aggregate_work(bytes + 2 * 29 + 21 + tokens);
        self.reserve(632, phase);
        self.aggregate_work(21 + bytes + tokens + 5);
        self.release(632);
        // No selected cut lies inside semantic verification; its bounded peak
        // is smaller than the earlier role scratch while the same owner lives.
        self.aggregate_work(
            semantic
                + bytes
                + bytes
                + 14
                + VERIFIED_CANONICAL_KERNEL_IR_V12_IDENTITY_DOMAIN_V1.len(),
        );
        self.release(retained);
        retained
    }
    fn custody(&mut self, profile: Profile, phase: &'static str) {
        use crate::{KirBridgeCorrespondenceV1, KirBridgeOptimizedReceiptV1};
        let (b, n, ba, r, u, e, payloads) = profile.shape();
        let (bytes, _, _, _) = profile.inverse();
        let tree = 6 + b + 2 * (n + b);
        let slots = ba + r + u;
        let parameter = usize::from(!matches!(profile, Profile::Empty));
        let source_types = ba - parameter + r;
        let volume = tree + slots + 1 + 2 * parameter + source_types + e + 1 + 2 * parameter;
        let envelope = 64 * (bytes + tree + slots + 1) + 4096;
        self.aggregate_work(1 + b + ba + 2 * (n + b) + r + u + e + payloads);
        self.work(bytes + tree, "custody.source.tree");
        self.aggregate_work(66 * (ba + r));
        self.work(
            4 * volume * volume + 8 * bytes * volume + 8 * (bytes + volume),
            "custody.envelope",
        );
        self.reserve(envelope, "custody.envelope");
        let retained = self.inverse(profile, phase);
        self.reserve(retained, "inverse.transfer");
        self.work(
            bytes + 12 + crate::KIR_PLIRON_BRIDGE_V12_IDENTITY_DOMAIN_V1.len(),
            "custody.digest",
        );
        let count = 1 + 2 * b + n;
        let capacity = count.max(4).next_power_of_two();
        let report = size_of::<KirBridgeOptimizedReceiptV1>()
            + capacity * size_of::<KirBridgeCorrespondenceV1>();
        self.reserve(report, "custody.receipt");
        self.release(envelope);
        self.work(
            2 * bytes,
            if phase == "custody.final.roles" {
                "custody.final.compare"
            } else {
                "custody.initial.compare"
            },
        );
        self.release(retained + report);
    }
    fn projection(&mut self, profile: Profile) {
        use crate::kir_bridge_v1::{
            NativeBridgeWitnessV1,
            canonical_trace_v1::{NativeFunctionV1, NativeOccurrenceV1},
        };
        use fe2o3_kernel_ir::{Type, ValueId};
        use pliron::{
            basic_block::BasicBlock, context::Ptr, operation::Operation, r#type::TypeHandle,
            value::Value,
        };
        let (b, n, ba, r, u, e, _) = profile.shape();
        let parameter = usize::from(!matches!(profile, Profile::Empty));
        let (bytes, _, _, _) = profile.inverse();
        let tree = 6 + b + 2 * (n + b);
        let slots = ba + r + u;
        let volume = tree + slots + 1 + 2 * parameter + ba - parameter + r + e + 1;
        self.reserve(size_of::<Projection<'_>>(), "projection.header");
        self.work(bytes, "import.source");
        self.work(
            4 * volume * volume + 8 * bytes * volume + 8 * (bytes + volume),
            "import.envelope",
        );
        self.reserve(64 * (bytes + tree + slots + 1) + 4096, "import.envelope");
        self.work(
            bytes + 12 + crate::KIR_PLIRON_BRIDGE_V12_IDENTITY_DOMAIN_V1.len(),
            "import.digest",
        );
        self.work(tree + 1, "import.witness");
        // SignatureRow has exactly these two field types in the pinned source.
        self.reserve(
            size_of::<NativeBridgeWitnessV1>() + size_of::<(Ptr<Operation>, TypeHandle)>(),
            "import.witness",
        );
        self.rows::<NativeFunctionV1<'_>>(1, "capture.functions");
        self.schema(&["sym_name"]);
        self.work(1, "capture.function");
        self.schema(&["sym_name", "func_type"]);
        self.work(8, "capture.function.name");
        self.work(1, "capture.census");
        self.aggregate_work(b - 1 + n);
        self.rows::<Ptr<BasicBlock>>(b, "capture.blocks");
        self.map::<Ptr<BasicBlock>, usize>(b, "capture.block.index");
        self.rows::<NativeOccurrenceV1<'_>>(n + b, "capture.occurrences");
        self.map::<Ptr<Operation>, usize>(n + b, "capture.occurrence.index");
        self.map::<Value, (ValueId, &Type)>(ba + r, "capture.values");
        for block in 0..b {
            self.work(1, "capture.block");
            self.aggregate_work(if block == 0 { parameter } else { 1 });
            let kinds: &[&str] = match profile {
                Profile::Empty => &[],
                Profile::Read => &["load"],
                Profile::Mixed => &["atomic", "barrier"],
                Profile::Later if block == 0 => &["constant"],
                Profile::Later => &["constant", "store"],
            };
            for kind in kinds {
                self.work(1, "capture.operation");
                self.schema(match *kind {
                    "load" => &[
                        "gpu_load_address_space",
                        "gpu_load_alignment",
                        "gpu_load_volatile",
                    ],
                    "store" => &[
                        "gpu_store_address_space",
                        "gpu_store_alignment",
                        "gpu_store_volatile",
                    ],
                    "constant" => &["gpu_constant_value"],
                    _ => &["gpu_preserved_operation_kind"],
                });
                if !matches!(*kind, "store" | "barrier") {
                    self.work(1, "capture.result");
                }
            }
            self.schema(&[]);
        }
        self.custody(profile, "custody.initial.roles");
    }
    fn scalar_query(&mut self, fold: &'static str) {
        use super::super::pliron_invocation_trace::native_values_v1::NativeScalarV1;
        use pliron::{
            attribute::AttrObj, builtin::attributes::IntegerAttr, utils::apint::APInt, value::Value,
        };
        let floor = self.live;
        self.work(1, "value.query");
        self.map::<Value, Option<NativeScalarV1>>(130, "value.cache");
        self.rows::<Value>(65, "value.active");
        self.work(1, "value.visit");
        self.work(0, "value.active.scan");
        self.work(2, fold);
        let scratch = 2 * size_of::<Vec<Option<AttrObj>>>()
            + size_of::<Option<AttrObj>>()
            + 3 * size_of::<IntegerAttr>()
            + 11 * size_of::<APInt>();
        self.reserve(scratch, fold);
        self.release(self.live - floor);
    }
    fn event(&mut self, name: &'static str) {
        self.work(1, name);
        if self.event_capacity < self.event_count_so_far() + 1 {
            let extra = self.event_capacity.max(1);
            self.reserve(
                extra * size_of::<super::super::pliron_invocation_trace::PlironTraceEventV1>(),
                name,
            );
            self.reserve(0, "event.capacity.reconciliation");
            self.event_capacity += extra;
        }
        self.steps.push(Step {
            label: "event.append",
            debit: Debit::Work(0),
            aggregate: false,
        });
    }
    fn event_count_so_far(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.label == "event.append")
            .count()
    }
    fn trace(&mut self, profile: Profile) {
        use super::super::pliron_invocation_trace::native_values_v1::NativeScalarV1;
        use pliron::value::Value;
        let (_, _, ba, _, _, _, _) = profile.shape();
        let environment = self.map::<Value, NativeScalarV1>(ba, "trace.environment");
        self.work(4, "trace.invocation");
        match profile {
            Profile::Empty => {}
            Profile::Read | Profile::Mixed => {
                self.work(1, "trace.operation");
                self.work(3, "event.fields");
                self.work(1, "event.address");
                self.event("event.first");
                if matches!(profile, Profile::Mixed) {
                    self.work(1, "trace.operation");
                    self.work(6, "barrier.mask");
                    self.event("event.later");
                }
            }
            Profile::Later => {
                self.work(1, "trace.operation");
                self.scalar_query("fold.first");
                self.work(1, "control.branch");
                let floor = self.live;
                self.rows::<(Value, Option<NativeScalarV1>)>(1, "control.payload");
                self.work(1, "control.source");
                self.scalar_query("fold.edge");
                self.work(1, "control.binding");
                self.release(self.live - floor);
                self.work(1, "trace.operation");
                self.scalar_query("fold.later");
                self.work(1, "trace.operation");
                self.work(3, "event.fields");
                self.work(1, "event.address");
                self.event("event.first");
            }
        }
        self.work(1, "control.return");
        self.work(0, "control.return.values");
        self.event("event.return");
        self.release(environment);
    }
    fn whole(profile: Profile) -> Self {
        use fe2o3_kernel_ir::FunctionId;
        let (b, n, _, _, _, _, _) = profile.shape();
        let mut out = Self::default();
        out.work(2, "protected.entry");
        out.work(1, "foundation.query");
        out.reserve(
            size_of::<Guard>()
                + size_of::<CheckedCanonicalInvocationTracesV1<'_, '_>>()
                + size_of::<std::thread::Result<Result<(), Failure>>>()
                + size_of::<std::thread::Result<()>>()
                + size_of::<Contract>(),
            "scope.headers",
        );
        out.projection(profile);
        out.rows::<RootRow>(1, "roots.rows");
        out.rows::<CanonicalNativeFunctionCensusV1>(1, "functions.rows");
        out.aggregate_work(1 + n + b);
        out.map::<&FunctionId, usize>(1, "functions.index");
        out.work(6, "functions.name");
        out.work(6, "roots.name");
        out.work(1, "root.projection");
        out.work(1, "input.identity");
        out.work(1, "input.geometry");
        out.aggregate_work(n + b);
        out.aggregate_work(b + n + b);
        out.trace(profile);
        out.work(1, "events.census");
        out.work(6 * profile.events() + 2, "barrier.participants");
        let scratch = 128 * profile.events() + 512;
        out.reserve(scratch, "barrier.participants");
        out.release(scratch);
        out.custody(profile, "custody.final.roles");
        out.work(1, "callback.query");
        out
    }
}

fn error_resource(error: &(dyn std::error::Error + 'static)) -> Option<Resource> {
    if let Some(resource) = error.downcast_ref::<Resource>() {
        return Some(*resource);
    }
    error.source().and_then(error_resource)
}

#[test]
fn bridge_resource_causes_remain_typed_through_native_scope_errors() {
    use crate::KirBridgeErrorV12 as BridgeError;
    use fe2o3_kernel_ir::CanonicalKernelIrReplayAdmissionErrorV12 as AdmissionError;
    use std::error::Error as _;

    let mut work = Work::new(0);
    let mut budget = Budget::new(&mut work, 0);
    let work_error = budget.charge_work(1).unwrap_err();
    let storage_error = budget.reserve_storage(1).unwrap_err();
    for (bridge, expected) in [
        (BridgeError::Resource(work_error), work_error),
        (
            BridgeError::Canonical(AdmissionError::Resource(storage_error)),
            storage_error,
        ),
        (
            BridgeError::Canonical(AdmissionError::Decode(
                fe2o3_kernel_ir::KernelIrDecodeError::Resource(storage_error),
            )),
            storage_error,
        ),
    ] {
        let error = CanonicalInvocationTraceErrorV1 {
            failure: Failure::Bridge(bridge),
        };
        assert_eq!(error_resource(&error), Some(expected));
    }
    let bridge = BridgeError::Bridge(crate::KirBridgeErrorV1::UnsupportedType);
    assert!(bridge.source().unwrap().is::<crate::KirBridgeErrorV1>());
    assert!(error_resource(&bridge).is_none());
    assert!(BridgeError::SessionSetup.source().is_none());
}

#[test]
fn whole_entry_empty_read_mixed_and_later_exact_and_one_short_are_predicted_before_execution() {
    for profile in [
        Profile::Empty,
        Profile::Read,
        Profile::Mixed,
        Profile::Later,
    ] {
        let oracle = Oracle::whole(profile);
        let full = oracle.predict(usize::MAX, usize::MAX);
        // Separate closed algebra in RESOURCE_DERIVATION.md, not oracle output.
        assert_eq!(
            full.work,
            match profile {
                Profile::Empty => 42_172,
                Profile::Read => 90_944,
                Profile::Mixed => 118_324,
                Profile::Later => 208_119,
            }
        );
        let mut work_limits = vec![full.work, full.work - 1, oracle.work_cut("capture.census")];
        if !matches!(profile, Profile::Empty) {
            work_limits.push(oracle.work_cut("event.first"));
        }
        if matches!(profile, Profile::Mixed) {
            work_limits.push(oracle.work_cut("event.later"));
        }
        if matches!(profile, Profile::Later) {
            for label in ["fold.first", "fold.edge", "fold.later", "control.binding"] {
                work_limits.push(oracle.work_cut(label));
            }
        }
        work_limits.push(oracle.work_cut("custody.final.compare"));
        for (work_limit, storage_limit) in work_limits
            .into_iter()
            .map(|work| (work, full.peak))
            .chain([(full.work, full.peak - 1)])
        {
            let predicted = oracle.predict(work_limit, storage_limit);
            tests::with_checked_space(&profile.source(), Some(storage_limit), |checked, budget| {
                let floor = budget.storage();
                let prefix = tests::WORK - work_limit;
                budget.charge_work(prefix - budget.work()).unwrap();
                let old_peak = budget.peak_storage();
                let result =
                    with_canonical_invocation_traces_v1(checked, budget, |view, budget| {
                        assert_eq!(budget.storage() - floor, full.retained);
                        assert_eq!(
                            view.attempt(0, budget)?,
                            CanonicalInvocationTraceAttemptV1::Complete {
                                invocations: 1,
                                events: profile.events()
                            }
                        );
                        Ok(())
                    });
                assert_eq!(
                    result.is_ok(),
                    predicted.denial.is_none(),
                    "{profile:?} {predicted:?}"
                );
                assert_eq!(
                    budget.work() - prefix,
                    predicted.work,
                    "{profile:?} {predicted:?}"
                );
                assert_eq!(budget.storage(), floor);
                assert_eq!(
                    budget.peak_storage(),
                    old_peak.max(floor + predicted.peak),
                    "{profile:?} {predicted:?}"
                );
                if let Some((_, work, attempted)) = predicted.denial {
                    let error = result.unwrap_err();
                    let resource = error_resource(&error).expect("typed first resource denial");
                    if work {
                        assert!(
                            matches!(resource, Resource::Work(error) if error.actual() == prefix + attempted && error.limit() == tests::WORK)
                        );
                    } else {
                        assert!(
                            matches!(resource, Resource::Storage(error) if error.actual() == floor + attempted && error.limit() == tests::STORAGE)
                        );
                        assert_eq!(budget.failed_storage(), Some(floor + attempted));
                    }
                }
            });
        }
    }
}

fn analysis_shape(profile: Profile) -> (Census, usize, usize, usize) {
    let (blocks, nonterminators, block_arguments, results, operands, successors, _) =
        profile.shape();
    let operations = nonterminators + blocks;
    let (attributes, arity) = match profile {
        Profile::Empty => (0, 0),
        Profile::Read => (3, 1),
        Profile::Mixed => (2, 1),
        Profile::Later => (5, 2),
    };
    let census = Census {
        blocks,
        operations,
        block_arguments,
        results,
        operands,
        successors,
        attributes,
        max_operation_arity: arity,
        ..Census::default()
    };
    // Existing logical-unit preflight: I=1, launch rank=3, rank ceiling=8,
    // steps=2^20, per visit=8+16+8, setup=3, evaluator scratch=1836.
    // max block arguments is one in Later, not its aggregate argument count 2.
    let steps = 1 << 20;
    let max_arguments = usize::from(!matches!(profile, Profile::Empty));
    let inventory = 2 * blocks + operations;
    // Nonempty entry arguments trigger a bounded scan of blocks, operations
    // and successor occurrences, even when Later ties the successor arity.
    let entry_scan = match profile {
        Profile::Empty => 0,
        Profile::Read | Profile::Mixed | Profile::Later => blocks + operations + successors + 1,
    };
    let work = 3 * blocks + 5 * operations + 5 + entry_scan + 36 * (steps + 1);
    // Read/Mixed have one entry argument, no other block and no entry edge:
    // at most one key is wide. Later has a one-argument successor, so it
    // retains the coarse bound; Empty has only zero-argument keys.
    let key_storage = match profile {
        Profile::Read | Profile::Mixed => 2 * steps + 1,
        Profile::Empty | Profile::Later => steps * (max_arguments + 2),
    };
    let retained = inventory + 4 + 17 * steps;
    let peak = retained + 2 * blocks + results + 3 * block_arguments + key_storage + 3 + 1836;
    (census, work, retained, peak)
}

#[test]
fn native_analysis_domain_exact_and_one_short_keep_legacy_caps_and_failed_cache() {
    for profile in [
        Profile::Empty,
        Profile::Read,
        Profile::Mixed,
        Profile::Later,
    ] {
        let (census, work, retained, peak) = analysis_shape(profile);
        let (owner, _) = tests::owner(&profile.source());
        let mut kir_work = Work::new(tests::WORK);
        let mut budget = Budget::new(&mut kir_work, tests::STORAGE);
        let projection = Projection::import(&owner, &mut budget).unwrap();
        projection
            .with_function(0, &mut budget, |context, function, row, budget| {
                let input =
                    NativeTraceInputV1::derive(&owner, context, row, 0, projection.epoch(), budget)
                        .unwrap();
                for (limit_work, limit_peak, accepted) in [
                    (work, peak, true),
                    (work - 1, peak, false),
                    (work, peak - 1, false),
                ] {
                    let floor = budget.storage();
                    let mut manager = Manager::new_with_resource_contract(
                        function,
                        census,
                        Bound::default(),
                        0,
                        Limits::new(limit_work, limit_peak),
                    )
                    .unwrap();
                    manager
                        .prepare_native_exact_trace_v1(context, function, &input, budget)
                        .unwrap();
                    assert_eq!(
                        manager.has_successful_exact_trace_v1(),
                        accepted,
                        "{profile:?}: work={limit_work}, peak={limit_peak}"
                    );
                    let bound = manager.resource_upper_bound();
                    if accepted {
                        assert_eq!(
                            (
                                bound.work_upper_bound(),
                                bound.retained_storage_upper_bound(),
                                bound.peak_storage_upper_bound()
                            ),
                            (work, retained, peak)
                        );
                    } else {
                        assert!(matches!(
                            manager.exact_trace(),
                            Err(TraceFailure::ResourceLimit)
                        ));
                        assert_eq!(
                            (
                                bound.work_upper_bound(),
                                bound.retained_storage_upper_bound()
                            ),
                            (
                                census.blocks + census.operations + 1,
                                2 * census.blocks + census.operations
                            )
                        );
                        manager
                            .prepare_native_exact_trace_v1(context, function, &input, budget)
                            .unwrap();
                        assert!(!manager.has_successful_exact_trace_v1());
                        assert_eq!(manager.resource_upper_bound(), bound);
                    }
                    drop(manager);
                    budget.release_storage(budget.storage() - floor).unwrap();
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn all_five_native_consumer_guards_have_exact_one_work_admission_and_sticky_denial() {
    let profile = Profile::Read;
    let (census, work, _, peak) = analysis_shape(profile);
    let (owner, _) = tests::owner(&profile.source());
    let mut kir_work = Work::new(tests::WORK);
    let mut budget = Budget::new(&mut kir_work, tests::STORAGE);
    let projection = Projection::import(&owner, &mut budget).unwrap();
    projection.with_function(0, &mut budget, |context, function, row, budget| {
        let input = NativeTraceInputV1::derive(&owner, context, row, 0, projection.epoch(), budget).unwrap();
        for consumer in 0..5 {
            for allowance in [0, 1] {
                let floor = budget.storage();
                let mut manager = Manager::new_with_resource_contract(function, census, Bound::default(), 0, Limits::new(work + allowance, peak)).unwrap();
                manager.prepare_native_exact_trace_v1(context, function, &input, budget).unwrap();
                assert!(manager.has_successful_exact_trace_v1());
                match consumer {
                    0 => { manager.prepare_provenance_alias(context, function); assert!(manager.provenance_alias().is_err()); }
                    1 => { manager.prepare_memory_order(context, function); assert!(manager.memory_order().is_err()); }
                    2 => assert!(!super::super::pliron_race::run_pliron_ranked_race_check_with_analyses_v1(context, function, &mut manager).is_clean()),
                    3 => assert!(!super::super::pliron_workgroup_memory::run_pliron_workgroup_memory_check_with_analyses_v1(context, function, &mut manager).is_clean()),
                    _ => assert!(!super::super::pliron_hierarchical_ownership::run_pliron_hierarchical_ownership_check_with_analyses_v1(context, function, &mut manager).is_clean()),
                }
                assert_eq!(manager.resource_upper_bound().work_upper_bound(), work + allowance);
                assert_eq!(manager.native_guard_denial_v1().is_some(), allowance == 0);
                assert!(manager.has_native_obligations_v1());
                let failure = manager.native_guard_denial_v1().unwrap();
                assert_eq!(failure.phase, Phase::InvocationTrace);
                assert_eq!(failure.resource, "work upper bound");
                let prefix = manager.resource_upper_bound();
                assert!(manager.has_native_obligations_v1());
                assert_eq!(manager.native_guard_denial_v1(), Some(failure));
                assert_eq!(manager.resource_upper_bound(), prefix);
                drop(manager);
                budget.release_storage(budget.storage() - floor).unwrap();
            }
        }
        Ok(())
    }).unwrap();
}
