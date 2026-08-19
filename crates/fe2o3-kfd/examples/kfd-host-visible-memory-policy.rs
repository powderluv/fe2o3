//! Default-feature linked fixture for the production memory authority surface.

use fe2o3_kfd::{
    DeviceSelector, HOST_VISIBLE_MEMORY_PROFILE_SHA256_V1, HostVisibleMemoryPhase, OpenedKfd,
};

fn parse_u64(value: &str) -> Result<u64, Box<dyn std::error::Error>> {
    if let Some(hex) = value.strip_prefix("0x") {
        Ok(u64::from_str_radix(hex, 16)?)
    } else {
        Ok(value.parse()?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let unique = std::env::args()
        .nth(1)
        .ok_or("usage: kfd-host-visible-memory-policy <selected-unique-id> [requested-bytes]")?;
    let requested = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(4097);
    let unique_id = parse_u64(&unique)?;
    let device = OpenedKfd::open_default()?
        .admit_uapi()?
        .bind_gfx942_xnack_minus(DeviceSelector::UniqueId(unique_id))?;
    let mut session = device.acquire_host_visible_memory_session()?;
    let layout = session.allocate(requested)?;
    session.with_bytes_mut(|bytes| bytes.fill(0x5a))?;
    session.with_bytes(|bytes| assert!(bytes.iter().all(|byte| *byte == 0x5a)))?;
    session.map_to_gpu()?;
    assert_eq!(session.phase(), HostVisibleMemoryPhase::GpuAccessible);
    session.unmap_from_gpu()?;
    session.with_bytes(|bytes| assert_eq!(bytes.first(), Some(&0x5a)))?;
    session.release()?;
    assert_eq!(session.phase(), HostVisibleMemoryPhase::Released);
    let model = session.model_journal_summary();
    println!(
        "profile_sha256={} unique_id={unique_id:016x} requested={} backing={} map_unmap=success release=success model_vms={} model_reservations={} model_allocations={} model_mappings={}",
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
