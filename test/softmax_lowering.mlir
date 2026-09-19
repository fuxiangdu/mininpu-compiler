module {
  func.func @last_axis(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %result = "mininpu.softmax"(%input) {axis = -1 : i64}
        : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }

  func.func @middle_axis(%input: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %result = "mininpu.softmax"(%input) {axis = 1 : i64}
        : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    return %result : tensor<2x3x4xf32>
  }
}
