module {
  func.func @dynamic_rms_norm(%input: tensor<?x16xf32>,
                              %weight: tensor<16xf32>)
      -> tensor<?x16xf32> {
    %result = "mininpu.rms_norm"(%input, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<?x16xf32>, tensor<16xf32>) -> tensor<?x16xf32>
    return %result : tensor<?x16xf32>
  }
}
