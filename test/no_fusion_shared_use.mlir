module {
  func.func @shared_matmul_result(
      %lhs: tensor<2x4xf32>,
      %rhs: tensor<4x8xf32>,
      %bias: tensor<8xf32>) -> (tensor<2x8xf32>, tensor<2x8xf32>) {
    %mm = "mininpu.matmul"(%lhs, %rhs)
        : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %biased = "mininpu.bias_add"(%mm, %bias)
        : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %result = "mininpu.relu"(%biased)
        : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %result, %mm : tensor<2x8xf32>, tensor<2x8xf32>
  }
}
