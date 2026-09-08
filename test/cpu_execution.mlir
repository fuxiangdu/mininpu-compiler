module attributes {
  mininpu.ub_bytes = 65536 : i64,
  mininpu.tile_granularity = 1 : i64
} {
  func.func private @check_f32(i32, f32, f32, f32) -> i32

  func.func @main() -> i32 {
    %lhs = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
        : tensor<2x2xf32>
    %rhs = arith.constant dense<[[1.0, -1.0], [2.0, 3.0]]>
        : tensor<2x2xf32>
    %bias = arith.constant dense<[-6.0, 1.0]> : tensor<2xf32>

    %mm = "mininpu.matmul"(%lhs, %rhs)
        : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %biased = "mininpu.bias_add"(%mm, %bias)
        : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    %result = "mininpu.relu"(%biased)
        : (tensor<2x2xf32>) -> tensor<2x2xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %id0 = arith.constant 0 : i32
    %id1 = arith.constant 1 : i32
    %id2 = arith.constant 2 : i32
    %id3 = arith.constant 3 : i32
    %expected0 = arith.constant 0.0 : f32
    %expected1 = arith.constant 6.0 : f32
    %expected2 = arith.constant 5.0 : f32
    %expected3 = arith.constant 10.0 : f32
    %atol = arith.constant 1.0e-5 : f32

    %actual0 = tensor.extract %result[%c0, %c0] : tensor<2x2xf32>
    %actual1 = tensor.extract %result[%c0, %c1] : tensor<2x2xf32>
    %actual2 = tensor.extract %result[%c1, %c0] : tensor<2x2xf32>
    %actual3 = tensor.extract %result[%c1, %c1] : tensor<2x2xf32>

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
