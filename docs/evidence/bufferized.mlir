#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {mininpu.tile_granularity = 16 : i64, mininpu.ub_bytes = 262144 : i64} {
  func.func private @consume_f32(f32)
  func.func @local_linear_relu(%arg0: memref<128x256xf32, strided<[?, ?], offset: ?>>, %arg1: memref<256x512xf32, strided<[?, ?], offset: ?>>, %arg2: memref<512xf32, strided<[?], offset: ?>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x512xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<128x512xf32>)
    linalg.matmul {mininpu.arithmetic_intensity = 4.1934477379095165 : f64, mininpu.double_buffered = true, mininpu.estimated_tile_count = 36 : i64, mininpu.estimated_working_set_bytes = 246144 : i64, mininpu.tile_k = 96 : i64, mininpu.tile_m = 112 : i64, mininpu.tile_n = 96 : i64, mininpu.ub_bytes = 262144 : i64} ins(%arg0, %arg1 : memref<128x256xf32, strided<[?, ?], offset: ?>>, memref<256x512xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<128x512xf32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<512xf32, strided<[?], offset: ?>>) outs(%alloc : memref<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %out, %in : f32
      %2 = arith.maximumf %1, %cst : f32
      linalg.yield %2 : f32
    }
    %0 = memref.load %alloc[%c0, %c0] : memref<128x512xf32>
    call @consume_f32(%0) : (f32) -> ()
    memref.dealloc %alloc : memref<128x512xf32>
    return
  }
}
