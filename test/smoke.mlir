module {
  func.func @identity(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %input : tensor<4x8xf32>
  }
}

