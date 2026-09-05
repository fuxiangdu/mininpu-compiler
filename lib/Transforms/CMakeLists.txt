add_mlir_library(MiniNPUTransforms
  FuseMatMulBiasRelu.cpp
  LowerToLinalg.cpp
  PlanTiles.cpp

  LINK_LIBS PUBLIC
  MiniNPUDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
