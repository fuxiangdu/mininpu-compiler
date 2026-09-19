module {
  func.func @invalid_axis(%input: tensor<2x4x16xf32>)
      -> tensor<2x4x16xf32> {
    %result = "mininpu.softmax"(%input) {axis = 3 : i64}
        : (tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    return %result : tensor<2x4x16xf32>
  }
}
