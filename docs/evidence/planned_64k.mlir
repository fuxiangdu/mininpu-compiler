module attributes {mininpu.tile_granularity = 16 : i64, mininpu.ub_bytes = 65536 : i64} {
  func.func @large_linear_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>) -> tensor<128x512xf32> {
    %0 = "mininpu.fused_matmul_bias_relu"(%arg0, %arg1, %arg2) {mininpu.arithmetic_intensity = 1.9930795847750864 : f64, mininpu.double_buffered = true, mininpu.estimated_tile_count = 198 : i64, mininpu.estimated_working_set_bytes = 55488 : i64, mininpu.tile_k = 48 : i64, mininpu.tile_m = 48 : i64, mininpu.tile_n = 48 : i64, mininpu.ub_bytes = 65536 : i64} : (tensor<128x256xf32>, tensor<256x512xf32>, tensor<512xf32>) -> tensor<128x512xf32>
    return %0 : tensor<128x512xf32>
  }
}

