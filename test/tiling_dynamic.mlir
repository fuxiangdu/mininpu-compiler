module {
  func.func @dynamic_shape(
      %lhs: tensor<?x256xf32>,
      %rhs: tensor<256x512xf32>,
      %bias: tensor<512xf32>) -> tensor<?x512xf32> {
    %result = "mininpu.fused_matmul_bias_relu"(%lhs, %rhs, %bias)
        : (tensor<?x256xf32>, tensor<256x512xf32>, tensor<512xf32>)
          -> tensor<?x512xf32>
    return %result : tensor<?x512xf32>
  }
}
