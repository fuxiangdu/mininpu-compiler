// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdint.h>
#include <stdio.h>

int32_t check_f32(int32_t index, float actual, float expected, float atol) {
  const float absolute_error = fabsf(actual - expected);
  const int32_t failed = !isfinite(actual) || absolute_error > atol;

  printf("[%s] output[%d] actual=%.8f expected=%.8f abs_error=%.3e\n",
         failed ? "FAIL" : "PASS", index, actual, expected,
         absolute_error);
  return failed;
}
