add_mlir_dialect_library(MiniNPUDialect
  MiniNPUDialect.cpp
  MiniNPUOps.cpp

  DEPENDS
  MiniNPUDialectIncGen
  MiniNPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
