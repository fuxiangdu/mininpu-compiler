module {
  func.func @invalid_epsilon(
      %input: tensor<2x4x16xf32>,
      %weight: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %result = "mininpu.rms_norm"(%input, %weight)
        {epsilon = 0.000000e+00 : f64}
        : (tensor<2x4x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %result : tensor<2x4x16xf32>
  }
}
