module attributes {mininpu.tile_granularity = 1 : i64, mininpu.ub_bytes = 65536 : i64} {
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<[-6.000000e+00, 1.000000e+00]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2x2xf32_0 : memref<2x2xf32> = dense<[[1.000000e+00, -1.000000e+00], [2.000000e+00, 3.000000e+00]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> {alignment = 64 : i64}
  func.func private @check_f32(i32, f32, f32, f32) -> i32
  func.func @main() -> i32 {
    %c2 = arith.constant 2 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 1.000000e+01 : f32
    %cst_1 = arith.constant 5.000000e+00 : f32
    %cst_2 = arith.constant 6.000000e+00 : f32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_2x2xf32 : memref<2x2xf32>
    %1 = memref.get_global @__constant_2x2xf32_0 : memref<2x2xf32>
    %2 = memref.get_global @__constant_2xf32 : memref<2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        memref.store %cst_3, %alloc[%arg0, %arg1] : memref<2x2xf32>
      }
    }
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        scf.for %arg2 = %c0 to %c2 step %c1 {
          %14 = memref.load %0[%arg0, %arg2] : memref<2x2xf32>
          %15 = memref.load %1[%arg2, %arg1] : memref<2x2xf32>
          %16 = memref.load %alloc[%arg0, %arg1] : memref<2x2xf32>
          %17 = arith.mulf %14, %15 : f32
          %18 = arith.addf %16, %17 : f32
          memref.store %18, %alloc[%arg0, %arg1] : memref<2x2xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        %14 = memref.load %2[%arg1] : memref<2xf32>
        %15 = memref.load %alloc[%arg0, %arg1] : memref<2x2xf32>
        %16 = arith.addf %15, %14 : f32
        %17 = arith.maximumf %16, %cst_3 : f32
        memref.store %17, %alloc[%arg0, %arg1] : memref<2x2xf32>
      }
    }
    %3 = memref.load %alloc[%c0, %c0] : memref<2x2xf32>
    %4 = memref.load %alloc[%c0, %c1] : memref<2x2xf32>
    %5 = memref.load %alloc[%c1, %c0] : memref<2x2xf32>
    %6 = memref.load %alloc[%c1, %c1] : memref<2x2xf32>
    %7 = call @check_f32(%c0_i32, %3, %cst_3, %cst) : (i32, f32, f32, f32) -> i32
    %8 = call @check_f32(%c1_i32, %4, %cst_2, %cst) : (i32, f32, f32, f32) -> i32
    %9 = call @check_f32(%c2_i32, %5, %cst_1, %cst) : (i32, f32, f32, f32) -> i32
    %10 = call @check_f32(%c3_i32, %6, %cst_0, %cst) : (i32, f32, f32, f32) -> i32
    %11 = arith.addi %7, %8 : i32
    %12 = arith.addi %9, %10 : i32
    %13 = arith.addi %11, %12 : i32
    memref.dealloc %alloc : memref<2x2xf32>
    return %13 : i32
  }
}
