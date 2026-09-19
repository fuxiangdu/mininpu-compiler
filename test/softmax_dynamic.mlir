module {
  func.func @dynamic_softmax(%input: tensor<?x16xf32>)
      -> tensor<?x16xf32> {
    %result = "mininpu.softmax"(%input) {axis = -1 : i64}
        : (tensor<?x16xf32>) -> tensor<?x16xf32>
    return %result : tensor<?x16xf32>
  }
}
