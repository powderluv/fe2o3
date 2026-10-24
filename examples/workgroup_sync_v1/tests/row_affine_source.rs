//! AST correspondence for the output-tile variant and the original SIMT body.

fn entry(source: &str) -> syn::ItemFn {
    syn::parse_file(source)
        .unwrap()
        .items
        .into_iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "row_affine_sum_u32_v1" => {
                Some(function)
            }
            _ => None,
        })
        .unwrap()
}

#[test]
fn output_tile_keeps_exact_admission_arithmetic_and_collective_ast() {
    let simt = entry(include_str!("../src/kernel_row_affine_u32.rs"));
    let tile = entry(include_str!("../src/kernel_row_affine_u32_tile.rs"));
    // Compare parsed syntax rather than trusting source hashes or marker strings.
    assert_eq!(simt.attrs, tile.attrs);
    assert_eq!(simt.sig.unsafety, None);
    assert_eq!(tile.sig.unsafety, None);
    assert_eq!(simt.sig.inputs.len(), 8);
    assert_eq!(tile.sig.inputs.len(), 8);
    assert_eq!(
        simt.sig.inputs.iter().take(7).collect::<Vec<_>>(),
        tile.sig.inputs.iter().take(7).collect::<Vec<_>>(),
    );
    let expected_output: syn::FnArg = syn::parse_quote!(
        mut output: DisjointSlice<u32, Tiled2D<Index1D, 64, 1, 64, 1>>
    );
    assert_eq!(tile.sig.inputs.last().unwrap(), &expected_output);
    let last = simt.block.stmts.len() - 1;
    assert_eq!(tile.block.stmts.len(), last + 1);
    assert_eq!(simt.block.stmts[..last], tile.block.stmts[..last]);
    let syn::Stmt::Expr(syn::Expr::If(simt_store), _) = &simt.block.stmts[last] else {
        panic!("SIMT store must be guarded");
    };
    let syn::Stmt::Expr(syn::Expr::If(tile_store), _) = &tile.block.stmts[last] else {
        panic!("tile store must be guarded");
    };
    let owner: syn::Expr = syn::parse_quote!(lane == 0 && row < rows);
    assert_eq!(*simt_store.cond, owner);
    assert_eq!(simt_store.cond, tile_store.cond);
    assert_eq!(tile_store.else_branch, None);
    let expected_store: syn::Block = syn::parse_quote!({
        let Some(tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            fe2o3_device::trap();
        };
        let Some(slot) = output.get_tiled_2d_mut(&tile, 0, rows, 1, 1) else {
            fe2o3_device::trap();
        };
        *slot = sum;
    });
    assert_eq!(tile_store.then_branch, expected_store);
}
