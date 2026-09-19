module {
  func.func private @check_f32(i32, f32, f32, f32) -> i32

  func.func @main() -> i32 {
    %query = arith.constant dense<[[[1.0, 0.0], [0.0, 1.0]]]>
        : tensor<1x2x2xf32>
    %key_transposed = arith.constant dense<[[[1.0, 0.0], [0.0, 1.0]]]>
        : tensor<1x2x2xf32>
    %value = arith.constant dense<[[[1.0, 2.0], [3.0, 4.0]]]>
        : tensor<1x2x2xf32>
    %weight = arith.constant dense<[1.0, 0.5]> : tensor<2xf32>

    %scores = "mininpu.batch_matmul"(%query, %key_transposed)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %probabilities = "mininpu.softmax"(%scores) {axis = -1 : i64}
        : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %context = "mininpu.batch_matmul"(%probabilities, %value)
        : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
    %result = "mininpu.rms_norm"(%context, %weight)
        {epsilon = 1.0e-5 : f64}
        : (tensor<1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %atol = arith.constant 1.0e-5 : f32
    %actual0 = tensor.extract %result[%c0, %c0, %c0]
        : tensor<1x2x2xf32>
    %actual1 = tensor.extract %result[%c0, %c0, %c1]
        : tensor<1x2x2xf32>
    %actual2 = tensor.extract %result[%c0, %c1, %c0]
        : tensor<1x2x2xf32>
    %actual3 = tensor.extract %result[%c0, %c1, %c1]
        : tensor<1x2x2xf32>

    %id0 = arith.constant 0 : i32
    %id1 = arith.constant 1 : i32
    %id2 = arith.constant 2 : i32
    %id3 = arith.constant 3 : i32
    %expected0 = arith.constant 0.73290902 : f32
    %expected1 = arith.constant 0.60473958 : f32
    %expected2 = arith.constant 0.81960691 : f32
    %expected3 = arith.constant 0.57624698 : f32

    %status0 = func.call @check_f32(%id0, %actual0, %expected0, %atol)
        : (i32, f32, f32, f32) -> i32
    %status1 = func.call @check_f32(%id1, %actual1, %expected1, %atol)
        : (i32, f32, f32, f32) -> i32
    %status2 = func.call @check_f32(%id2, %actual2, %expected2, %atol)
        : (i32, f32, f32, f32) -> i32
    %status3 = func.call @check_f32(%id3, %actual3, %expected3, %atol)
        : (i32, f32, f32, f32) -> i32

    %status01 = arith.addi %status0, %status1 : i32
    %status23 = arith.addi %status2, %status3 : i32
    %status = arith.addi %status01, %status23 : i32
    return %status : i32
  }
}
