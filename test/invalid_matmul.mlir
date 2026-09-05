module {
  func.func @invalid_matmul(
      %lhs: tensor<2x3xf32>,
      %rhs: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %result = "mininpu.matmul"(%lhs, %rhs)
        : (tensor<2x3xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    return %result : tensor<2x8xf32>
  }
}
