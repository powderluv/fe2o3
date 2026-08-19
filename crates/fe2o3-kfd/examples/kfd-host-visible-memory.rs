use std::process::Command;

use fe2o3_kfd::{
    DeviceSelector, HOST_VISIBLE_MEMORY_PROFILE_SHA256_V1, HostVisibleMemoryPhase, OpenedKfd,
};

const CHILD_ENV: &str = "FE2O3_KFD_MEMORY_LIVE_CHILD";

fn parse_u64(value: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Some(hex) = value.strip_prefix("0x") {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(value.parse()?)
    }
}

fn run_child(unique_id: u64, requested: usize) -> Result<(), Box<dyn std::error::Error>> {
    let device = OpenedKfd::open_default()?
        .admit_uapi()?
        .bind_gfx942_xnack_minus(DeviceSelector::UniqueId(unique_id))?;
    let mut session = device.acquire_host_visible_memory_session()?;
    let layout = session.allocate(requested)?;
    session.with_bytes_mut(|bytes| {
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(17).wrapping_add(3);
        }
    })?;
    session.verify_dontfork_child_negative()?;
    session.with_bytes(|bytes| {
        assert_eq!(bytes[0], 3);
        assert_eq!(
            bytes[bytes.len() - 1],
            ((bytes.len() - 1) as u8).wrapping_mul(17).wrapping_add(3)
        );
    })?;
    session.map_to_gpu()?;
    assert_eq!(session.phase(), HostVisibleMemoryPhase::GpuAccessible);
    session.unmap_from_gpu()?;
    session.with_bytes(|bytes| assert_eq!(bytes[0], 3))?;
    session.release()?;
    assert_eq!(session.phase(), HostVisibleMemoryPhase::Released);
    let model = session.model_journal_summary();
    println!(
        "profile_sha256={} unique_id={unique_id:016x} requested={} backing={} dontfork_child=absent map_unmap=success release=success model_vms={} model_reservations={} model_allocations={} model_mappings={}",
        HOST_VISIBLE_MEMORY_PROFILE_SHA256_V1,
        layout.requested_bytes(),
        layout.backing_bytes(),
        model.vm_records(),
        model.reservation_records(),
        model.allocation_records(),
        model.mapping_records(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let unique = std::env::args()
        .nth(1)
        .ok_or("usage: kfd-host-visible-memory <selected-unique-id> [requested-bytes]")?;
    let requested = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(4097);
    if std::env::var_os(CHILD_ENV).is_some() {
        return run_child(parse_u64(&unique)?, requested);
    }

    let status = Command::new(std::env::current_exe()?)
        .arg(unique)
        .arg(requested.to_string())
        .env(CHILD_ENV, "1")
        .status()?;
    if !status.success() {
        return Err(format!("isolated memory child failed with {status}").into());
    }
    Ok(())
}
