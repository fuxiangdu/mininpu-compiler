module {
  func.func @dynamic_batch(
      %lhs: tensor<?x4x8xf32>,
      %rhs: tensor<?x8x16xf32>) -> tensor<?x4x16xf32> {
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<?x4x8xf32>, tensor<?x8x16xf32>) -> tensor<?x4x16xf32>
    return %result : tensor<?x4x16xf32>
  }
}
