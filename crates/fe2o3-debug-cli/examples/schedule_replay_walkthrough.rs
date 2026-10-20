//! Bounded CPU debugger walkthrough over a strictly admitted Bundle V6.
//! Usage: schedule_replay_walkthrough BUNDLE REQUEST ORACLE_JSON [SAMPLES=5]
use fe2o3_kir_debugger::{
    DebugInspectionV1, DebugNavigationV1, DebugScopeSelectorV1, DebugSessionV1,
    DebugTranscriptCompletenessV1, DebugWaveWidthV1, DebuggerLimitsV1,
    capture_debugger_replayed_run_v1,
};
use fe2o3_kir_sim::{
    PersistedSimulationScheduleDocumentV1, SimulationDebugCaptureLimitsV1,
    SimulationDebugRecordKindV1, SimulationExecutionV1, SimulationScheduleRequestV1,
};
use fe2o3_kir_sim_cli::{load_debug_sidecar_v1, load_debug_simulation_bundle_v6};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{error::Error, io, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MAX_BUFFER: usize = 4096;
const ROUNDTRIPS: usize = 128;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Oracle {
    buffer_argument: usize,
    shared_backing: Option<u32>,
    allocation: u64,
    initial_bytes_hex: String,
    initial_initialized_hex: String,
    bytes_hex: String,
    initialized_hex: String,
}

fn fail(message: &str) -> Box<dyn Error> {
    io::Error::other(message).into()
}
fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(fail(message))
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn decode_hex(value: &str) -> Result<Vec<u8>> {
    ensure(
        !value.is_empty() && value.len() <= MAX_BUFFER * 2 && value.len() % 2 == 0,
        "oracle hex must encode 1..=4096 bytes",
    )?;
    ensure(
        value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "oracle hex must use lowercase hexadecimal",
    )?;
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair)?, 16).map_err(Into::into))
        .collect()
}
fn initialized(value: &str, bytes: usize) -> Result<Vec<bool>> {
    let bits = decode_hex(value)?;
    ensure(
        bits.len() == bytes.div_ceil(8),
        "oracle initialization bitmap length",
    )?;
    if bytes % 8 != 0 {
        ensure(
            bits[bits.len() - 1] >> (bytes % 8) == 0,
            "oracle bitmap padding must be zero",
        )?;
    }
    Ok((0..bytes)
        .map(|i| bits[i / 8] & (1 << (i % 8)) != 0)
        .collect())
}
fn check_output(
    execution: &SimulationExecutionV1,
    oracle: &Oracle,
    bytes: &[u8],
    init: &[bool],
) -> Result<()> {
    let output = match oracle.shared_backing {
        Some(id) => execution.shared_buffer(fe2o3_kir_sim::BufferBackingIdV1(id)),
        None => execution.buffer(oracle.buffer_argument),
    }
    .ok_or_else(|| fail("oracle buffer argument or shared backing missing"))?;
    ensure(
        output.bytes() == bytes,
        "independent output byte oracle mismatch",
    )?;
    ensure(
        output.initialized() == init,
        "independent output initialization oracle mismatch",
    )
}
fn check_memory(
    session: &DebugSessionV1,
    oracle: &Oracle,
    bytes: &[u8],
    init: &[bool],
) -> Result<()> {
    let DebugInspectionV1::Available(memory) = session.memory(oracle.allocation, 0, bytes.len())
    else {
        return Err(fail("selected checkpoint memory is unavailable"));
    };
    ensure(
        memory.address_space == fe2o3_kernel_ir::AddressSpace::Global,
        "oracle allocation is not Global",
    )?;
    ensure(memory.bytes == bytes, "checkpoint byte oracle mismatch")?;
    ensure(
        memory.initialized == init,
        "checkpoint initialization oracle mismatch",
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure(
        (3..=4).contains(&args.len()),
        "usage: schedule_replay_walkthrough BUNDLE REQUEST ORACLE_JSON [SAMPLES=5]",
    )?;
    let samples = args
        .get(3)
        .map(|s| s.to_str().ok_or_else(|| fail("sample count must be UTF-8")))
        .transpose()?
        .unwrap_or("5")
        .parse::<usize>()?;
    ensure(
        (1..=16).contains(&samples),
        "sample count must be in 1..=16",
    )?;
    let oracle_bytes =
        load_debug_sidecar_v1(Path::new(&args[2]), 32 * 1024).map_err(|e| fail(&e.message))?;
    let oracle: Oracle = serde_json::from_slice(&oracle_bytes)?;
    ensure(oracle.allocation != 0, "allocation must be nonzero")?;
    let expected = decode_hex(&oracle.bytes_hex)?;
    let expected_init = initialized(&oracle.initialized_hex, expected.len())?;
    let initial = decode_hex(&oracle.initial_bytes_hex)?;
    ensure(
        initial.len() == expected.len(),
        "initial and final oracle lengths must match",
    )?;
    let initial_init = initialized(&oracle.initial_initialized_hex, initial.len())?;

    let start = Instant::now();
    let owner = load_debug_simulation_bundle_v6(Path::new(&args[0]), Path::new(&args[1]))
        .map_err(|e| fail(&e.message))?;
    let input = owner.input();
    let admission_ns = start.elapsed().as_nanos();
    let binding = input
        .persisted_schedule_binding()
        .map_err(|e| fail(&e.message))?;
    let start = Instant::now();
    let recorded = input
        .module
        .simulate_scheduled(
            &input.request,
            input.simulation_target(),
            input.simulation_limits,
            SimulationScheduleRequestV1::RecordSeeded {
                seed: 0x5eed,
                max_decisions: 4096,
            },
        )
        .map_err(|e| fail(&format!("schedule recording: {e:?}")))?;
    let record_ns = start.elapsed().as_nanos();
    check_output(&recorded, &oracle, &expected, &expected_init)?;
    let schedule_bytes = PersistedSimulationScheduleDocumentV1::encode_record(
        binding,
        recorded
            .schedule_record()
            .ok_or_else(|| fail("schedule record missing"))?,
    )
    .map_err(|e| fail(&format!("schedule encoding: {e:?}")))?;
    let document = PersistedSimulationScheduleDocumentV1::from_canonical_bytes(&schedule_bytes)
        .map_err(|e| fail(&format!("schedule decoding: {e:?}")))?;
    ensure(
        document.binding() == binding,
        "persisted schedule binding changed",
    )?;

    let mut capture_ns = Vec::new();
    let mut navigation_ns = Vec::new();
    let mut retained_records = 0;
    let mut first_checkpoint = 0;
    let mut last_checkpoint = 0;
    // One warmup, then a closed sample count. Every iteration checks correctness.
    for sample in 0..=samples {
        let start = Instant::now();
        let run = capture_debugger_replayed_run_v1(
            &input.module,
            &input.request,
            input.simulation_target(),
            input.simulation_limits,
            SimulationDebugCaptureLimitsV1::new(16, 256, 16, 8192).unwrap(),
            DebuggerLimitsV1::new(8192, 65_536, 16 << 20).unwrap(),
            DebugWaveWidthV1::Wave64,
            document.record(),
        );
        let elapsed = start.elapsed().as_nanos();
        let execution = run
            .execution
            .as_ref()
            .map_err(|e| fail(&format!("debugger replay: {e:?}")))?;
        check_output(execution, &oracle, &expected, &expected_init)?;
        ensure(
            execution.arguments() == recorded.arguments()
                && execution.shared_buffers() == recorded.shared_buffers(),
            "replay outputs changed",
        )?;
        ensure(
            execution.schedule_transcript_identity() == recorded.schedule_transcript_identity(),
            "replay schedule transcript identity changed",
        )?;
        ensure(
            run.transcript.completeness() == DebugTranscriptCompletenessV1::Complete,
            "debugger transcript was truncated",
        )?;
        let checkpoints: Vec<_> = run
            .transcript
            .records()
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                matches!(r.kind, SimulationDebugRecordKindV1::Checkpoint { .. }).then_some(i)
            })
            .collect();
        let first = *checkpoints.first().ok_or_else(|| fail("no checkpoints"))?;
        let last = *checkpoints.last().unwrap();
        ensure(last > first, "walkthrough needs at least two checkpoints")?;
        if sample != 0 {
            ensure(
                run.transcript.records().len() == retained_records
                    && first == first_checkpoint
                    && last == last_checkpoint,
                "repeated capture structure changed",
            )?;
        }
        retained_records = run.transcript.records().len();
        first_checkpoint = first;
        last_checkpoint = last;
        let mut session = DebugSessionV1::new(run.transcript);
        ensure(
            matches!(
                session.seek_record_index(first),
                DebugNavigationV1::Stopped(_)
            ),
            "initial seek failed",
        )?;
        check_memory(&session, &oracle, &initial, &initial_init)?;
        ensure(
            matches!(
                session.seek_record_index(last),
                DebugNavigationV1::Stopped(_)
            ),
            "final seek failed",
        )?;
        check_memory(&session, &oracle, &expected, &expected_init)?;
        // Seeking backward observes the retained initial snapshot; it does not undo execution.
        session.seek_record_index(first);
        check_memory(&session, &oracle, &initial, &initial_init)?;
        session.seek_record_index(last);
        let start = Instant::now();
        for _ in 0..ROUNDTRIPS {
            ensure(
                matches!(
                    session.reverse_step(&DebugScopeSelectorV1::Dispatch),
                    DebugNavigationV1::Stopped(_)
                ),
                "reverse step failed",
            )?;
            ensure(
                session.cursor_record_index().is_some_and(|i| i < last),
                "reverse cursor did not move",
            )?;
            ensure(
                matches!(
                    session.forward_step(&DebugScopeSelectorV1::Dispatch),
                    DebugNavigationV1::Stopped(_)
                ),
                "forward step failed",
            )?;
            ensure(
                session.cursor_record_index() == Some(last),
                "forward step did not restore cursor",
            )?;
        }
        let nav = start.elapsed().as_nanos();
        check_memory(&session, &oracle, &expected, &expected_init)?;
        if sample != 0 {
            capture_ns.push(elapsed);
            navigation_ns.push(nav);
        }
    }
    let report = serde_json::json!({
        "schema":"fe2o3-debugger-replay-walkthrough-v1", "status":"pass",
        "scope":"CPU library debugger; retained bundle; immutable transcript navigation",
        "bundle_sha256":hex(&Sha256::digest(owner.bundle().canonical_bytes())),
        "request_sha256":hex(&input.request_sha256), "oracle_sha256":hex(&Sha256::digest(&oracle_bytes)),
        "kir_sha256":hex(input.module.identity().digest()),
        "kir_bytes":input.module.identity().canonical_length(),
        "schedule_sha256":hex(&Sha256::digest(&schedule_bytes)),
        "schedule_decisions":document.record().decisions().len(),
        "output_bytes":expected.len(), "output_hex":hex(&expected),
        "initialized_bytes":expected_init.iter().filter(|x| **x).count(),
        "records":retained_records, "first_checkpoint":first_checkpoint, "last_checkpoint":last_checkpoint,
        "samples":samples, "warmup_captures":1, "roundtrips_per_sample":ROUNDTRIPS,
        "admission_ns":admission_ns, "record_schedule_ns":record_ns,
        "capture_replay_ns":capture_ns, "forward_reverse_ns":navigation_ns,
        "hardware_observed":false, "source_authenticated":false
    });
    serde_json::to_writer(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oracle_hex_is_bounded_and_canonical() {
        assert_eq!(decode_hex("0080ff").unwrap(), [0, 128, 255]);
        for s in ["", "0", "FF", "gg"] {
            assert!(decode_hex(s).is_err());
        }
        assert!(decode_hex(&"00".repeat(MAX_BUFFER + 1)).is_err());
    }
    #[test]
    fn initialization_masks_are_little_bit_first_with_zero_padding() {
        assert_eq!(initialized("05", 3).unwrap(), [true, false, true]);
        assert!(initialized("ff", 3).is_err());
        assert!(initialized("0000", 3).is_err());
    }
}
