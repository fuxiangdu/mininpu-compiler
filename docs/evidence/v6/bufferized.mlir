#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {mininpu.tile_granularity = 1 : i64, mininpu.ub_bytes = 65536 : i64} {
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<[-6.000000e+00, 1.000000e+00]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2x2xf32_0 : memref<2x2xf32> = dense<[[1.000000e+00, -1.000000e+00], [2.000000e+00, 3.000000e+00]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> {alignment = 64 : i64}
  func.func private @check_f32(i32, f32, f32, f32) -> i32
  func.func @main() -> i32 {
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
    linalg.fill ins(%cst_3 : f32) outs(%alloc : memref<2x2xf32>)
    linalg.matmul {mininpu.arithmetic_intensity = 0.076923076923076927 : f64, mininpu.double_buffered = true, mininpu.estimated_tile_count = 1 : i64, mininpu.estimated_working_set_bytes = 104 : i64, mininpu.tile_k = 2 : i64, mininpu.tile_m = 2 : i64, mininpu.tile_n = 2 : i64, mininpu.ub_bytes = 65536 : i64} ins(%0, %1 : memref<2x2xf32>, memref<2x2xf32>) outs(%alloc : memref<2x2xf32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<2xf32>) outs(%alloc : memref<2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.addf %out, %in : f32
      %15 = arith.maximumf %14, %cst_3 : f32
      linalg.yield %15 : f32
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
