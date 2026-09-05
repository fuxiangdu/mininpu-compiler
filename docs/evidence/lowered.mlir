#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {mininpu.tile_granularity = 16 : i64, mininpu.ub_bytes = 262144 : i64} {
  func.func @large_linear_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>) -> tensor<128x512xf32> {
    %0 = tensor.empty() : tensor<128x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %2 = linalg.matmul {mininpu.arithmetic_intensity = 4.1934477379095165 : f64, mininpu.double_buffered = true, mininpu.estimated_tile_count = 36 : i64, mininpu.estimated_working_set_bytes = 246144 : i64, mininpu.tile_k = 96 : i64, mininpu.tile_m = 112 : i64, mininpu.tile_n = 96 : i64, mininpu.ub_bytes = 262144 : i64} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%2 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %out, %in : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %5 = arith.maximumf %4, %cst_0 : f32
      linalg.yield %5 : f32
    } -> tensor<128x512xf32>
    return %3 : tensor<128x512xf32>
  }
}

