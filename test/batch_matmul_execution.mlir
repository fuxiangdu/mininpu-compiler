module {
  func.func private @check_f32(i32, f32, f32, f32) -> i32

  func.func @main() -> i32 {
    %lhs = arith.constant dense<[
      [[1.0, 2.0], [3.0, 4.0]],
      [[-1.0, 0.0], [2.0, 1.0]]
    ]> : tensor<2x2x2xf32>
    %rhs = arith.constant dense<[[[1.0, -1.0], [2.0, 3.0]]]>
        : tensor<1x2x2xf32>
    %result = "mininpu.batch_matmul"(%lhs, %rhs)
        : (tensor<2x2x2xf32>, tensor<1x2x2xf32>) -> tensor<2x2x2xf32>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %atol = arith.constant 1.0e-5 : f32

    %actual0 = tensor.extract %result[%c0, %c0, %c0]
        : tensor<2x2x2xf32>
    %actual1 = tensor.extract %result[%c0, %c0, %c1]
        : tensor<2x2x2xf32>
    %actual2 = tensor.extract %result[%c0, %c1, %c0]
        : tensor<2x2x2xf32>
    %actual3 = tensor.extract %result[%c0, %c1, %c1]
        : tensor<2x2x2xf32>
    %actual4 = tensor.extract %result[%c1, %c0, %c0]
        : tensor<2x2x2xf32>
    %actual5 = tensor.extract %result[%c1, %c0, %c1]
        : tensor<2x2x2xf32>
    %actual6 = tensor.extract %result[%c1, %c1, %c0]
        : tensor<2x2x2xf32>
    %actual7 = tensor.extract %result[%c1, %c1, %c1]
        : tensor<2x2x2xf32>

    %id0 = arith.constant 0 : i32
    %id1 = arith.constant 1 : i32
    %id2 = arith.constant 2 : i32
    %id3 = arith.constant 3 : i32
    %id4 = arith.constant 4 : i32
    %id5 = arith.constant 5 : i32
    %id6 = arith.constant 6 : i32
    %id7 = arith.constant 7 : i32
    %expected0 = arith.constant 5.0 : f32
    %expected1 = arith.constant 5.0 : f32
    %expected2 = arith.constant 11.0 : f32
    %expected3 = arith.constant 9.0 : f32
    %expected4 = arith.constant -1.0 : f32
    %expected5 = arith.constant 1.0 : f32
    %expected6 = arith.constant 4.0 : f32
    %expected7 = arith.constant 1.0 : f32

    %status0 = func.call @check_f32(%id0, %actual0, %expected0, %atol)
        : (i32, f32, f32, f32) -> i32
    %status1 = func.call @check_f32(%id1, %actual1, %expected1, %atol)
        : (i32, f32, f32, f32) -> i32
    %status2 = func.call @check_f32(%id2, %actual2, %expected2, %atol)
        : (i32, f32, f32, f32) -> i32
    %status3 = func.call @check_f32(%id3, %actual3, %expected3, %atol)
        : (i32, f32, f32, f32) -> i32
    %status4 = func.call @check_f32(%id4, %actual4, %expected4, %atol)
        : (i32, f32, f32, f32) -> i32
    %status5 = func.call @check_f32(%id5, %actual5, %expected5, %atol)
        : (i32, f32, f32, f32) -> i32
    %status6 = func.call @check_f32(%id6, %actual6, %expected6, %atol)
        : (i32, f32, f32, f32) -> i32
    %status7 = func.call @check_f32(%id7, %actual7, %expected7, %atol)
        : (i32, f32, f32, f32) -> i32

    %status01 = arith.addi %status0, %status1 : i32
    %status23 = arith.addi %status2, %status3 : i32
    %status45 = arith.addi %status4, %status5 : i32
    %status67 = arith.addi %status6, %status7 : i32
    %status03 = arith.addi %status01, %status23 : i32
    %status47 = arith.addi %status45, %status67 : i32
    %status = arith.addi %status03, %status47 : i32
    return %status : i32
  }
}
