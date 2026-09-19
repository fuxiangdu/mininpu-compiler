module {
  func.func @invalid_output_batch(
      %lhs: tensor<1x4x8xf32>,
      %rhs: tensor<3x8x16xf32>) -> tensor<2x4x16xf32> {
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<1x4x8xf32>, tensor<3x8x16xf32>) -> tensor<2x4x16xf32>
    return %result : tensor<2x4x16xf32>
  }
}
