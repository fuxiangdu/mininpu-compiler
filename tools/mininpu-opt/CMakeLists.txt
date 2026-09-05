add_llvm_executable(mininpu-opt
  mininpu-opt.cpp
)

llvm_update_compile_flags(mininpu-opt)

target_link_libraries(mininpu-opt
  PRIVATE
  MiniNPUTransforms
  MiniNPUDialect
  MLIRMlirOptMain
)

mlir_check_all_link_libraries(mininpu-opt)
