module attributes {
  mininpu.ub_bytes = 262144 : i64,
  mininpu.tile_granularity = 16 : i64
} {
  // The scalar sink keeps one element of the result observable while leaving
  // the tensor allocation local to this function.  This lets the regression
  // test verify both function-boundary bufferization and ownership-based
  // deallocation without depending on a runtime library.
  func.func private @consume_f32(f32)

  func.func @local_linear_relu(
      %lhs: tensor<128x256xf32>,
      %rhs: tensor<256x512xf32>,
      %bias: tensor<512xf32>) {
    %mm = "mininpu.matmul"(%lhs, %rhs)
        : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    %biased = "mininpu.bias_add"(%mm, %bias)
        : (tensor<128x512xf32>, tensor<512xf32>) -> tensor<128x512xf32>
    %result = "mininpu.relu"(%biased)
        : (tensor<128x512xf32>) -> tensor<128x512xf32>
    %c0 = arith.constant 0 : index
    %sample = tensor.extract %result[%c0, %c0]
        : tensor<128x512xf32>
    func.call @consume_f32(%sample) : (f32) -> ()
    return
  }
}
