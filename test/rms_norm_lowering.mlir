module {
  func.func @rank_one(%input: tensor<4xf32>, %weight: tensor<4xf32>)
      -> tensor<4xf32> {
    %result = "mininpu.rms_norm"(%input, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func @rank_two(%input: tensor<2x4xf32>, %weight: tensor<4xf32>)
      -> tensor<2x4xf32> {
    %result = "mininpu.rms_norm"(%input, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @rank_three_f16(%input: tensor<2x3x4xf16>,
                            %weight: tensor<4xf16>)
      -> tensor<2x3x4xf16> {
    %result = "mininpu.rms_norm"(%input, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<2x3x4xf16>, tensor<4xf16>) -> tensor<2x3x4xf16>
    return %result : tensor<2x3x4xf16>
  }
}
