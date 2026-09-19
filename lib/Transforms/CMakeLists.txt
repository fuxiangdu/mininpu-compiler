add_mlir_library(MiniNPUTransforms
  ApplyTiles.cpp
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
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
