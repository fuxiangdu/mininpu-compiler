module {
  func.func @partial_plan(
      %lhs: tensor<2x4xf32>,
      %rhs: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %empty = tensor.empty() : tensor<2x8xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<2x8xf32>) -> tensor<2x8xf32>
    %result = linalg.matmul {mininpu.tile_m = 2 : i64}
        ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x8xf32>)
        outs(%init : tensor<2x8xf32>) -> tensor<2x8xf32>
    return %result : tensor<2x8xf32>
  }
}
