use fe2o3_kfd::topology::discover_default_topology;
use fe2o3_kfd::{GFX942_QUEUE_RESOURCE_PROFILE_SHA256_V1, plan_gfx942_aql_queue_resources};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = discover_default_topology()?;
    for gpu in snapshot.topology().gpu_nodes() {
        let plan = plan_gfx942_aql_queue_resources(&snapshot, gpu.unique_id(), 4096)?;
        println!(
            "authority=none profile_sha256={} node={} gpu_id={} unique_id={:016x} generation={} mes={} sched_policy={} cwsr_enable={} ring={} control_page={} eop={} cwsr={} ctl_per_xcc={} ctx_per_xcc={} debug_per_xcc={} xcc={} doorbell_width={} doorbell_slice={} backing=reviewed-rocr-expression",
            GFX942_QUEUE_RESOURCE_PROFILE_SHA256_V1,
            gpu.node_id(),
            plan.target().gpu_id(),
            plan.target().unique_id(),
            plan.target().topology_generation(),
            plan.target().mes(),
            plan.target().sched_policy(),
            plan.target().cwsr_enable(),
            plan.ring().mapping_bytes(),
            plan.control().exact_mapping_bytes_per_pointer(),
            plan.end_of_pipe().mapping_bytes(),
            plan.context_save().mapping_bytes(),
            plan.context_save().control_stack_bytes_per_xcc(),
            plan.context_save().context_save_bytes_per_xcc(),
            plan.context_save().debug_bytes_per_xcc(),
            plan.context_save().xcc_count(),
            plan.doorbell().width_bytes(),
            plan.doorbell().process_slice_bytes(),
        );
    }
    Ok(())
}
