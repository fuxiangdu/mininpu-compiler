module {
  func.func @invalid_element_type(%input: tensor<2x4xi32>)
      -> tensor<2x4xi32> {
    %result = "mininpu.softmax"(%input) {axis = -1 : i64}
        : (tensor<2x4xi32>) -> tensor<2x4xi32>
    return %result : tensor<2x4xi32>
  }
}
