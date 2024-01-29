module attributes { transform.with_named_sequence } {
  transform.named_sequence @codegen(%variant_op: !transform.any_op {transform.consumed}) {
    // Get matmul op
    // ==========================================
    %matmul = transform.structured.match ops{["linalg.matmul_transpose_b"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.print %variant_op : !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_matmul, %forall_grid = transform.structured.tile_using_forall %matmul tile_sizes [256, 128]
    ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Fuse fill
    // ==========================================
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    transform.structured.fuse_into_containing_op %fill into %forall_grid :
    (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func0 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func0 : !transform.any_op

    // Tile fill
    // ==========================================
    %fill2 = transform.structured.match ops{["linalg.fill"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %tiled_fill3, %forall3 = transform.structured.tile_using_forall %fill2 tile_sizes [64, 64] (mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile reduction dim
    // ==========================================
    %loop, %tiled_matmul2 = transform.structured.tile_using_for %tiled_matmul [0, 0, 32] :  (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote lhs and rhs
    // ==========================================
    %promoted_matmul, %alloc_a, %alloc_b = transform.iree.promote_operands %tiled_matmul2 [0, 1]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile to warps
    // ==========================================
    %tiled_matmul3, %forall2 = transform.structured.tile_using_forall %promoted_matmul tile_sizes [64, 64] (mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    //transform.apply_cse to %func0 : !transform.any_op

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    // %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    // transform.apply_patterns to %func_2 {
    //   transform.apply_patterns.iree.fold_extf_into_contraction
    // } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [64, 8, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %func_7 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %func_7 : !transform.any_op
    transform.apply_patterns to %func_7 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_7 : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    transform.yield
  }
}
