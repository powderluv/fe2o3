//! One numerical completion inside the existing cooperative interpreter.
use super::*;
use crate::matrix_bf16_exact_v1::{self as domain, MatrixInputRoleV1};
use fe2o3_kernel_ir::F32MathFunction;

pub(super) const ARRIVAL_WORK: usize = 24;
const RESOLUTION_FIXED_WORK: usize = 21_824;

#[derive(Clone, Copy)]
pub(super) struct Input {
    lhs: [u16; 4],
    rhs: [u16; 4],
    accumulator: [u32; 4],
}

struct Scratch {
    members: [usize; 64],
    results: [[u32; 4]; 64],
}

/// Logical payload, not an RSS/whole-host-stack claim. Reserve before execution.
pub(super) fn resident_bytes() -> Option<usize> {
    // Allow construction/move coexistence for the fixed scratch and result array.
    size_of::<Scratch>()
        .checked_mul(2)?
        .checked_add(size_of::<[RuntimeValue; 4]>().checked_mul(2)?)?
        .checked_add(size_of::<[ScalarBitsV1; 6]>())?
        .checked_add(size_of::<Input>().checked_mul(2)?)?
        .checked_add(size_of::<[u32; 3]>())
}

pub(super) fn resolution_work(participants: usize) -> Option<usize> {
    participants
        .checked_mul(66)?
        .checked_add(RESOLUTION_FIXED_WORK)
}

// 768 selected-domain checks, 4096 FMA-result finiteness checks, mode dispatch.
// Prepaid by the existing caller before rendezvous resolution, never a cap raise.
pub(super) fn resolution_work_with_numerics(
    participants: usize,
    mode: crate::MatrixBf16NumericsV1,
) -> Option<usize> {
    resolution_work(participants)?.checked_add(match mode {
        crate::MatrixBf16NumericsV1::ExactV1 => 0,
        crate::MatrixBf16NumericsV1::FiniteSequentialFmaV1 => 768 + 4096 + 1,
    })
}

pub(super) fn is_operation(operation: &OperationKind) -> bool {
    matches!(operation, OperationKind::Matrix(matrix) if domain::supported(matrix))
}

pub(super) fn prepare(
    engine: &mut Engine<'_, impl SimulationEventSinkV1>,
    frame: &RuntimeFrame<'_>,
    matrix: &MatrixOperation,
    site: CompactSite,
) -> Result<Input, SimulationExecutionErrorV1> {
    if !domain::supported(matrix) {
        return Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant("unadmitted matrix exact profile"),
        ));
    }
    let MatrixOperationKind::MultiplyAccumulate {
        lhs,
        rhs,
        accumulator,
        ..
    } = &matrix.kind
    else {
        return Err(engine.at(
            site,
            SimulationExecutionErrorKindV1::InternalInvariant("matrix exact kind"),
        ));
    };
    engine.charge_steps(&site, ARRIVAL_WORK)?;
    let mut input = Input {
        lhs: [0; 4],
        rhs: [0; 4],
        accumulator: [0; 4],
    };
    for (ids, output) in [(lhs, &mut input.lhs), (rhs, &mut input.rhs)] {
        for (id, slot) in ids.iter().zip(output) {
            let scalar = scalar_value(engine, &frame.values, *id, &site)?;
            if scalar.ty() != ScalarType::Bf16 {
                return Err(engine.at(
                    site,
                    SimulationExecutionErrorKindV1::RuntimeType {
                        value: Some(*id),
                        expected: "BF16 matrix input",
                    },
                ));
            }
            *slot = scalar.bits() as u16;
        }
    }
    for (id, slot) in accumulator.iter().zip(&mut input.accumulator) {
        let scalar = scalar_value(engine, &frame.values, *id, &site)?;
        if scalar.ty() != ScalarType::F32 {
            return Err(engine.at(
                site,
                SimulationExecutionErrorKindV1::RuntimeType {
                    value: Some(*id),
                    expected: "F32 matrix accumulator",
                },
            ));
        }
        *slot = scalar.bits() as u32;
    }
    Ok(input)
}

fn input<'m>(machine: &'m InvocationMachine<'_>) -> Option<&'m Input> {
    match machine.collective_input()? {
        CollectiveInput::MatrixBf16Exact(input) => Some(input),
        _ => None,
    }
}

fn f32_value(bits: u32, target: SimulationTargetV1) -> ScalarBitsV1 {
    // A u32 always fits the fixed F32 scalar width, independent of index width.
    ScalarBitsV1::new(ScalarType::F32, u128::from(bits), target)
        .expect("u32 is an exact F32 bit container")
}

/// Called only after the existing full-wave/site rendezvous and work precharge.
/// Every operand is validated and every output calculated before any binding.
pub(super) fn resolve<'a>(
    engine: &mut Engine<'a, impl SimulationEventSinkV1>,
    machines: &mut [InvocationMachine<'a>],
    arrival: CollectiveArrival<'a>,
    start: u64,
) -> Result<(), SimulationExecutionErrorV1> {
    let mut scratch = Scratch {
        members: [usize::MAX; 64],
        results: [[0; 4]; 64],
    };
    for (index, machine) in machines.iter().enumerate() {
        let local = local_linear(machine.invocation);
        if let Some(lane) = local.checked_sub(start).filter(|lane| *lane < 64) {
            let slot = &mut scratch.members[lane as usize];
            if *slot != usize::MAX {
                return Err(engine.at(
                    arrival.site,
                    SimulationExecutionErrorKindV1::InternalInvariant(
                        "matrix duplicate wave member",
                    ),
                ));
            }
            *slot = index;
        }
    }
    // Do not short-circuit the prepaid 768-component scan on a domain failure.
    let mut rejected = None;
    for (lane, index) in scratch.members.iter().copied().enumerate() {
        let machine = machines.get(index).ok_or_else(|| {
            engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::InternalInvariant("matrix missing wave member"),
            )
        })?;
        let values = input(machine).ok_or_else(|| {
            engine.at(
                arrival.site,
                SimulationExecutionErrorKindV1::InternalInvariant("matrix missing typed input"),
            )
        })?;
        for (role, values) in [
            (MatrixInputRoleV1::A, &values.lhs),
            (MatrixInputRoleV1::B, &values.rhs),
        ] {
            for (component, bits) in values.iter().copied().enumerate() {
                if !engine.matrix_bf16_numerics.accepts_bf16(bits) && rejected.is_none() {
                    rejected = Some((lane, role, component));
                }
            }
        }
        for (component, bits) in values.accumulator.iter().copied().enumerate() {
            if !engine.matrix_bf16_numerics.accepts_accumulator(bits) && rejected.is_none() {
                rejected = Some((lane, MatrixInputRoleV1::Accumulator, component));
            }
        }
    }
    if let Some((lane, role, component)) = rejected {
        engine.select_debug_invocation(Some(machines[scratch.members[lane]].invocation));
        return Err(engine.at(
            arrival.site,
            SimulationExecutionErrorKindV1::UnsupportedMatrixInputDomain {
                role,
                lane: lane as u8,
                component: component as u8,
            },
        ));
    }
    for (lane, output) in scratch.results.iter_mut().enumerate() {
        let column = lane % 16;
        let accumulator =
            input(&machines[scratch.members[lane]]).expect("all typed wave inputs checked");
        for (component, result) in output.iter_mut().enumerate() {
            let row = 4 * (lane / 16) + component;
            let mut value = f32_value(accumulator.accumulator[component], engine.target);
            for k in 0..16 {
                let a = input(&machines[scratch.members[row + 16 * (k / 4)]])
                    .expect("all typed A inputs checked")
                    .lhs[k % 4];
                let b = input(&machines[scratch.members[column + 16 * (k / 4)]])
                    .expect("all typed B inputs checked")
                    .rhs[k % 4];
                let operands = [
                    f32_value(u32::from(a) << 16, engine.target),
                    f32_value(u32::from(b) << 16, engine.target),
                    value,
                ];
                value = crate::soft_float::execute_compact_operation_v1(
                    SoftFloatOperationV1::F32Math(F32MathFunction::FusedMultiplyAdd),
                    &operands,
                    engine.target,
                )
                .map_err(|error| engine.at(arrival.site, map_soft_float_error(error)))?;
                if engine.matrix_bf16_numerics == crate::MatrixBf16NumericsV1::FiniteSequentialFmaV1
                    && !domain::finite_f32(value.bits() as u32)
                {
                    engine
                        .select_debug_invocation(Some(machines[scratch.members[lane]].invocation));
                    return Err(engine.at(
                        arrival.site,
                        SimulationExecutionErrorKindV1::NonFiniteMatrixResult {
                            lane: lane as u8,
                            component: component as u8,
                            reduction: k as u8,
                        },
                    ));
                }
            }
            *result = value.bits() as u32;
        }
    }
    for (lane, index) in scratch.members.iter().copied().enumerate() {
        let results =
            scratch.results[lane].map(|bits| RuntimeValue::Scalar(f32_value(bits, engine.target)));
        engine.select_debug_invocation(Some(machines[index].invocation));
        machines[index].complete_collective(engine, &results)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "execute_matrix_bf16_exact_v1_tests.rs"]
mod tests;
