module {
  func.func @constant_fold() -> i32 {
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %sum = arith.addi %c7, %c8 : i32
    return %sum : i32
  }
}

