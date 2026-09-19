module {
  func.func @invalid_batch(
      %lhs: tensor<2x4x8xf32>,
      %rhs: tensor<3x8x16xf32>) -> tensor<3x4x16xf32> {
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<2x4x8xf32>, tensor<3x8x16xf32>) -> tensor<3x4x16xf32>
    return %result : tensor<3x4x16xf32>
  }
}
