module {
  func.func @streamable(%query: tensor<1x2x2xf32>,
                        %key_transposed: tensor<1x2x2xf32>,
                        %value: tensor<1x2x2xf32>,
                        %weight: tensor<2xf32>) -> tensor<1x2x2xf32> {
    %scores = "mininpu.batch_matmul"(%query, %key_transposed)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %probabilities = "mininpu.softmax"(%scores) {axis = -1 : i64}
        : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %context = "mininpu.batch_matmul"(%probabilities, %value)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %normalized = "mininpu.rms_norm"(%context, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2xf32>
    return %normalized : tensor<1x2x2xf32>
  }

  func.func @shared_scores(%query: tensor<1x2x2xf32>,
                           %key_transposed: tensor<1x2x2xf32>,
                           %value: tensor<1x2x2xf32>,
                           %weight: tensor<2xf32>)
      -> (tensor<1x2x2xf32>, tensor<1x2x2xf32>) {
    %scores = "mininpu.batch_matmul"(%query, %key_transposed)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %probabilities = "mininpu.softmax"(%scores) {axis = -1 : i64}
        : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %context = "mininpu.batch_matmul"(%probabilities, %value)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %normalized = "mininpu.rms_norm"(%context, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2xf32>
    return %normalized, %scores : tensor<1x2x2xf32>, tensor<1x2x2xf32>
  }

  // Softmax over a non-final dimension is not an attention probability edge.
  func.func @non_last_axis(%query: tensor<1x2x2xf32>,
                           %key_transposed: tensor<1x2x2xf32>,
                           %value: tensor<1x2x2xf32>,
                           %weight: tensor<2xf32>) -> tensor<1x2x2xf32> {
    %scores = "mininpu.batch_matmul"(%query, %key_transposed)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %probabilities = "mininpu.softmax"(%scores) {axis = 1 : i64}
        : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %context = "mininpu.batch_matmul"(%probabilities, %value)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %normalized = "mininpu.rms_norm"(%context, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2xf32>
    return %normalized : tensor<1x2x2xf32>
  }
}
