// Checked output memory tiles; the original SIMT/Verus body stays separate.
macro_rules! vecadd_tile_body {
    ($thread:ident, ($($arg:expr),* $(,)?), $add:ident, $a:ident, $b:ident, $output:ident) => {{
        let index = $thread::index_1d($($arg),*);
        let i = index.get();
        let extent = $output.len();
        let Some(tile) = index.checked_tiled_2d::<64, 1, 64, 1>() else {
            return;
        };
        if let Some(out) = $output.get_tiled_2d_mut(&tile, 0, 1, extent, extent) {
            *out = $add!($a[i], $b[i]);
        }
    }};
}
