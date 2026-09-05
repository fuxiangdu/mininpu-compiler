set(LLVM_TARGET_DEFINITIONS MiniNPUDialect.td)
mlir_tablegen(MiniNPUDialect.h.inc -gen-dialect-decls -dialect=mininpu)
mlir_tablegen(MiniNPUDialect.cpp.inc -gen-dialect-defs -dialect=mininpu)
add_public_tablegen_target(MiniNPUDialectIncGen)

set(LLVM_TARGET_DEFINITIONS MiniNPUOps.td)
mlir_tablegen(MiniNPUOps.h.inc -gen-op-decls)
mlir_tablegen(MiniNPUOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MiniNPUOpsIncGen)
