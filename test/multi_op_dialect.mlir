module {
  func.func @multi_op_graph(
      %lhs: tensor<2x4x8xf32>,
      %rhs: tensor<1x8x16xf32>,
      %weight: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %batched = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<2x4x8xf32>, tensor<1x8x16xf32>) -> tensor<2x4x16xf32>
    %probabilities = "mininpu.softmax"(%batched) {axis = -1 : i64}
        : (tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    %normalized = "mininpu.rms_norm"(%probabilities, %weight)
        {epsilon = 1.000000e-06 : f64}
        : (tensor<2x4x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %normalized : tensor<2x4x16xf32>
  }
}
