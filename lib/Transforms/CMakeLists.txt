add_mlir_library(MiniNPUTransforms
  ApplyTiles.cpp
  FuseMatMulBiasRelu.cpp
  LowerToLinalg.cpp
  PlanAttention.cpp
  PlanTiles.cpp

  LINK_LIBS PUBLIC
  MiniNPUDialect
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
)
