//! Structured output tiles for the existing qwen-ngram-gather kernel.
#![allow(missing_docs)]
use crate::{NGRAM, QUERIES, SYSTEM_BATCHES, TABLE_SIZE};
use fe2o3_device::{DisjointSlice, Index1D, Tiled2D, kernel, thread};

/// Probes every slot, verifies the full 3-gram, and resolves duplicate keys.
#[cfg(feature = "kernel-qwen-ngram-gather-tile")]
#[cfg_attr(
    not(feature = "ablation-ngram-reverse-probe"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-ngram-reverse-probe",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
pub fn gfx950_qwen_ngram_gather_v1(
    queries: &[i32],
    table_hashes: &[u64],
    table_grams: &[i32],
    table_values: &[i32],
    priorities: &[i32],
    mut output: DisjointSlice<i32, Tiled2D<Index1D, 64, 1, 64, 1>>,
) {
    // Each in-range lane owns one query; no subgroup coordination is required.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let query = global_index & 63;
    if batch >= SYSTEM_BATCHES
        || queries.len() != SYSTEM_BATCHES * QUERIES * NGRAM
        || table_hashes.len() != SYSTEM_BATCHES * TABLE_SIZE
        || table_grams.len() != SYSTEM_BATCHES * TABLE_SIZE * NGRAM
        || table_values.len() != SYSTEM_BATCHES * TABLE_SIZE
        || priorities.len() != SYSTEM_BATCHES * TABLE_SIZE
        || output.len() != SYSTEM_BATCHES * QUERIES
    {
        return;
    }
    if query >= QUERIES {
        return;
    }
    // Hash all three tokens; the full gram is still checked to reject collisions.
    let query_batch_base = batch.wrapping_mul(QUERIES).wrapping_mul(NGRAM);
    let table_batch_base = batch.wrapping_mul(TABLE_SIZE);
    let gram_batch_base = batch.wrapping_mul(TABLE_SIZE).wrapping_mul(NGRAM);
    let base = query_batch_base.wrapping_add(query.wrapping_mul(NGRAM));
    let mut hash = 1_469_598_103_934_665_603_u64;
    hash ^= queries[base] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    hash ^= queries[base.wrapping_add(1)] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    hash ^= queries[base.wrapping_add(2)] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    let mut best_slot = usize::MAX;
    let mut best_priority = i32::MIN;
    let mut best_value = -1_i32;
    // Probe the bounded table completely and resolve duplicate keys deterministically.
    macro_rules! probe {
        ($probe:literal) => {{
            let slot = hash.wrapping_add($probe) as usize & (TABLE_SIZE - 1);
            let table_slot = table_batch_base.wrapping_add(slot);
            let gram_slot = gram_batch_base.wrapping_add(slot.wrapping_mul(NGRAM));
            let equal = (table_hashes[table_slot] == hash)
                & (table_grams[gram_slot] == queries[base])
                & (table_grams[gram_slot.wrapping_add(1)] == queries[base.wrapping_add(1)])
                & (table_grams[gram_slot.wrapping_add(2)] == queries[base.wrapping_add(2)]);
            if equal {
                let priority = priorities[table_slot];
                if priority > best_priority || (priority == best_priority && slot < best_slot) {
                    best_slot = slot;
                    best_priority = priority;
                    best_value = table_values[table_slot];
                }
            }
        }};
    }
    #[cfg(not(feature = "ablation-ngram-reverse-probe"))]
    macro_rules! final_probe {
        ($probe:literal) => {{
            let slot = hash.wrapping_add($probe) as usize & (TABLE_SIZE - 1);
            let table_slot = table_batch_base.wrapping_add(slot);
            let gram_slot = gram_batch_base.wrapping_add(slot.wrapping_mul(NGRAM));
            let equal = (table_hashes[table_slot] == hash)
                & (table_grams[gram_slot] == queries[base])
                & (table_grams[gram_slot.wrapping_add(1)] == queries[base.wrapping_add(1)])
                & (table_grams[gram_slot.wrapping_add(2)] == queries[base.wrapping_add(2)]);
            if equal {
                let priority = priorities[table_slot];
                if priority > best_priority || (priority == best_priority && slot < best_slot) {
                    best_value = table_values[table_slot];
                }
            }
        }};
    }
    #[cfg(not(feature = "ablation-ngram-reverse-probe"))]
    {
        probe!(0);
        probe!(1);
        probe!(2);
        probe!(3);
        probe!(4);
        probe!(5);
        probe!(6);
        probe!(7);
        probe!(8);
        probe!(9);
        probe!(10);
        probe!(11);
        probe!(12);
        probe!(13);
        probe!(14);
        final_probe!(15);
    }
    #[cfg(feature = "ablation-ngram-reverse-probe")]
    {
        probe!(15);
        probe!(14);
        probe!(13);
        probe!(12);
        probe!(11);
        probe!(10);
        probe!(9);
        probe!(8);
        probe!(7);
        probe!(6);
        probe!(5);
        probe!(4);
        probe!(3);
        probe!(2);
        probe!(1);
        probe!(0);
    }
    // A checked 1x64 tile gives each query lane its complete output coordinate.
    let Some(output_tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
        return;
    };
    if let Some(slot) = output.get_tiled_2d_mut(&output_tile, 0, SYSTEM_BATCHES, QUERIES, QUERIES) {
        *slot = best_value;
    }
}
