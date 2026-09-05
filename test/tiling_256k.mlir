module attributes {
  mininpu.ub_bytes = 262144 : i64,
  mininpu.tile_granularity = 16 : i64
} {
  func.func @large_linear_relu(
      %lhs: tensor<128x256xf32>,
      %rhs: tensor<256x512xf32>,
      %bias: tensor<512xf32>) -> tensor<128x512xf32> {
    %mm = "mininpu.matmul"(%lhs, %rhs)
        : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    %biased = "mininpu.bias_add"(%mm, %bias)
        : (tensor<128x512xf32>, tensor<512xf32>) -> tensor<128x512xf32>
    %result = "mininpu.relu"(%biased)
        : (tensor<128x512xf32>) -> tensor<128x512xf32>
    return %result : tensor<128x512xf32>
  }
}
