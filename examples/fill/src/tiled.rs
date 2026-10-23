use fe2o3_device::{DisjointSlice, Index1D, Tiled2D, kernel, thread};

/// Each 64-lane workgroup owns a 64x1 row-major output tile.
#[kernel(typed, launch(required = [64, 1, 1], max = [64, 1, 1]))]
pub fn fill(mut out: DisjointSlice<f32, Tiled2D<Index1D, 64, 64, 1, 1>>) {
    let rows = out.len();
    let Some(tile) = thread::index_1d().checked_tiled_2d::<64, 64, 1, 1>() else {
        return;
    };
    let Some(value) = out.get_tiled_2d_mut(&tile, 0, rows, 1, 1) else {
        return;
    };
    *value = 42.5;
}
