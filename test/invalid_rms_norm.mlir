module {
  func.func @invalid_weight(
      %input: tensor<2x4x16xf32>,
      %weight: tensor<8xf32>) -> tensor<2x4x16xf32> {
    %result = "mininpu.rms_norm"(%input, %weight)
        {epsilon = 1.000000e-06 : f64}
        : (tensor<2x4x16xf32>, tensor<8xf32>) -> tensor<2x4x16xf32>
    return %result : tensor<2x4x16xf32>
  }
}
