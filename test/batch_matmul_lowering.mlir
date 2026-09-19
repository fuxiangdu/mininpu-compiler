module {
  func.func @equal_batch(
      %lhs: tensor<2x4x8xf32>,
      %rhs: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    return %result : tensor<2x4x16xf32>
  }

  func.func @broadcast_lhs_batch(
      %lhs: tensor<1x4x8xf32>,
      %rhs: tensor<3x8x16xf32>) -> tensor<3x4x16xf32> {
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<1x4x8xf32>, tensor<3x8x16xf32>) -> tensor<3x4x16xf32>
    return %result : tensor<3x4x16xf32>
  }

  func.func @broadcast_rhs_batch(
      %lhs: tensor<3x4x8xf32>,
      %rhs: tensor<1x8x16xf32>) -> tensor<3x4x16xf32> {
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<3x4x8xf32>, tensor<1x8x16xf32>) -> tensor<3x4x16xf32>
    return %result : tensor<3x4x16xf32>
  }
}
