module attributes {mininpu.tile_granularity = 1 : i64, mininpu.ub_bytes = 65536 : i64} {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_2xf32(dense<[-6.000000e+00, 1.000000e+00]> : tensor<2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x f32>
  llvm.mlir.global private constant @__constant_2x2xf32_0(dense<[[1.000000e+00, -1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x array<2 x f32>>
  llvm.mlir.global private constant @__constant_2x2xf32(dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x array<2 x f32>>
  llvm.func @check_f32(i32, f32, f32, f32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %3 = llvm.mlir.constant(1.000000e+01 : f32) : f32
    %4 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(6.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(3 : i32) : i32
    %8 = llvm.mlir.constant(2 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(0 : i32) : i32
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(2 : index) : i64
    %13 = llvm.mlir.constant(2 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(4 : index) : i64
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.addressof @__constant_2x2xf32 : !llvm.ptr
    %20 = llvm.getelementptr %19[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x f32>>
    %21 = llvm.mlir.constant(3735928559 : index) : i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %22, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.insertvalue %20, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %28 = llvm.insertvalue %12, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.insertvalue %13, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %13, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %14, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.mlir.constant(2 : index) : i64
    %33 = llvm.mlir.constant(2 : index) : i64
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.mlir.constant(4 : index) : i64
    %36 = llvm.mlir.zero : !llvm.ptr
    %37 = llvm.getelementptr %36[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.mlir.addressof @__constant_2x2xf32_0 : !llvm.ptr
    %40 = llvm.getelementptr %39[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x f32>>
    %41 = llvm.mlir.constant(3735928559 : index) : i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    %43 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.insertvalue %42, %43[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.insertvalue %40, %44[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(0 : index) : i64
    %47 = llvm.insertvalue %46, %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.insertvalue %32, %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.insertvalue %33, %48[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.insertvalue %33, %49[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.insertvalue %34, %50[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.mlir.constant(2 : index) : i64
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.mlir.zero : !llvm.ptr
    %55 = llvm.getelementptr %54[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.addressof @__constant_2xf32 : !llvm.ptr
    %58 = llvm.getelementptr %57[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x f32>
    %59 = llvm.mlir.constant(3735928559 : index) : i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %61 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %62 = llvm.insertvalue %60, %61[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %63 = llvm.insertvalue %58, %62[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %64 = llvm.mlir.constant(0 : index) : i64
    %65 = llvm.insertvalue %64, %63[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %66 = llvm.insertvalue %52, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %67 = llvm.insertvalue %53, %66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %68 = llvm.mlir.constant(2 : index) : i64
    %69 = llvm.mlir.constant(2 : index) : i64
    %70 = llvm.mlir.constant(1 : index) : i64
    %71 = llvm.mlir.constant(4 : index) : i64
    %72 = llvm.mlir.zero : !llvm.ptr
    %73 = llvm.getelementptr %72[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %74 = llvm.ptrtoint %73 : !llvm.ptr to i64
    %75 = llvm.mlir.constant(64 : index) : i64
    %76 = llvm.add %74, %75  : i64
    %77 = llvm.call @malloc(%76) : (i64) -> !llvm.ptr
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.sub %75, %79  : i64
    %81 = llvm.add %78, %80  : i64
    %82 = llvm.urem %81, %75  : i64
    %83 = llvm.sub %81, %82  : i64
    %84 = llvm.inttoptr %83 : i64 to !llvm.ptr
    %85 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %86 = llvm.insertvalue %77, %85[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %88 = llvm.mlir.constant(0 : index) : i64
    %89 = llvm.insertvalue %88, %87[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %90 = llvm.insertvalue %68, %89[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %91 = llvm.insertvalue %69, %90[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %92 = llvm.insertvalue %69, %91[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.insertvalue %70, %92[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%0 : i64)
  ^bb1(%94: i64):  // 2 preds: ^bb0, ^bb5
    %95 = llvm.icmp "slt" %94, %1 : i64
    llvm.cond_br %95, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0 : i64)
  ^bb3(%96: i64):  // 2 preds: ^bb2, ^bb4
    %97 = llvm.icmp "slt" %96, %1 : i64
    llvm.cond_br %97, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %98 = llvm.mlir.constant(2 : index) : i64
    %99 = llvm.mul %94, %98  : i64
    %100 = llvm.add %99, %96  : i64
    %101 = llvm.getelementptr %84[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %6, %101 : f32, !llvm.ptr
    %102 = llvm.add %96, %11  : i64
    llvm.br ^bb3(%102 : i64)
  ^bb5:  // pred: ^bb3
    %103 = llvm.add %94, %11  : i64
    llvm.br ^bb1(%103 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%0 : i64)
  ^bb7(%104: i64):  // 2 preds: ^bb6, ^bb14
    %105 = llvm.icmp "slt" %104, %1 : i64
    llvm.cond_br %105, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%0 : i64)
  ^bb9(%106: i64):  // 2 preds: ^bb8, ^bb13
    %107 = llvm.icmp "slt" %106, %1 : i64
    llvm.cond_br %107, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%0 : i64)
  ^bb11(%108: i64):  // 2 preds: ^bb10, ^bb12
    %109 = llvm.icmp "slt" %108, %1 : i64
    llvm.cond_br %109, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %110 = llvm.mlir.constant(2 : index) : i64
    %111 = llvm.mul %104, %110  : i64
    %112 = llvm.add %111, %108  : i64
    %113 = llvm.getelementptr %20[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %114 = llvm.load %113 : !llvm.ptr -> f32
    %115 = llvm.mlir.constant(2 : index) : i64
    %116 = llvm.mul %108, %115  : i64
    %117 = llvm.add %116, %106  : i64
    %118 = llvm.getelementptr %40[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %119 = llvm.load %118 : !llvm.ptr -> f32
    %120 = llvm.mlir.constant(2 : index) : i64
    %121 = llvm.mul %104, %120  : i64
    %122 = llvm.add %121, %106  : i64
    %123 = llvm.getelementptr %84[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.load %123 : !llvm.ptr -> f32
    %125 = llvm.fmul %114, %119  : f32
    %126 = llvm.fadd %124, %125  : f32
    %127 = llvm.mlir.constant(2 : index) : i64
    %128 = llvm.mul %104, %127  : i64
    %129 = llvm.add %128, %106  : i64
    %130 = llvm.getelementptr %84[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %126, %130 : f32, !llvm.ptr
    %131 = llvm.add %108, %11  : i64
    llvm.br ^bb11(%131 : i64)
  ^bb13:  // pred: ^bb11
    %132 = llvm.add %106, %11  : i64
    llvm.br ^bb9(%132 : i64)
  ^bb14:  // pred: ^bb9
    %133 = llvm.add %104, %11  : i64
    llvm.br ^bb7(%133 : i64)
  ^bb15:  // pred: ^bb7
    llvm.br ^bb16(%0 : i64)
  ^bb16(%134: i64):  // 2 preds: ^bb15, ^bb20
    %135 = llvm.icmp "slt" %134, %1 : i64
    llvm.cond_br %135, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%0 : i64)
  ^bb18(%136: i64):  // 2 preds: ^bb17, ^bb19
    %137 = llvm.icmp "slt" %136, %1 : i64
    llvm.cond_br %137, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %138 = llvm.getelementptr %58[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %139 = llvm.load %138 : !llvm.ptr -> f32
    %140 = llvm.mlir.constant(2 : index) : i64
    %141 = llvm.mul %134, %140  : i64
    %142 = llvm.add %141, %136  : i64
    %143 = llvm.getelementptr %84[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %144 = llvm.load %143 : !llvm.ptr -> f32
    %145 = llvm.fadd %144, %139  : f32
    %146 = llvm.intr.maximum(%145, %6)  : (f32, f32) -> f32
    %147 = llvm.mlir.constant(2 : index) : i64
    %148 = llvm.mul %134, %147  : i64
    %149 = llvm.add %148, %136  : i64
    %150 = llvm.getelementptr %84[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %146, %150 : f32, !llvm.ptr
    %151 = llvm.add %136, %11  : i64
    llvm.br ^bb18(%151 : i64)
  ^bb20:  // pred: ^bb18
    %152 = llvm.add %134, %11  : i64
    llvm.br ^bb16(%152 : i64)
  ^bb21:  // pred: ^bb16
    %153 = llvm.mlir.constant(2 : index) : i64
    %154 = llvm.mul %0, %153  : i64
    %155 = llvm.add %154, %0  : i64
    %156 = llvm.getelementptr %84[%155] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %157 = llvm.load %156 : !llvm.ptr -> f32
    %158 = llvm.mlir.constant(2 : index) : i64
    %159 = llvm.mul %0, %158  : i64
    %160 = llvm.add %159, %11  : i64
    %161 = llvm.getelementptr %84[%160] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %162 = llvm.load %161 : !llvm.ptr -> f32
    %163 = llvm.mlir.constant(2 : index) : i64
    %164 = llvm.mul %11, %163  : i64
    %165 = llvm.add %164, %0  : i64
    %166 = llvm.getelementptr %84[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %167 = llvm.load %166 : !llvm.ptr -> f32
    %168 = llvm.mlir.constant(2 : index) : i64
    %169 = llvm.mul %11, %168  : i64
    %170 = llvm.add %169, %11  : i64
    %171 = llvm.getelementptr %84[%170] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %172 = llvm.load %171 : !llvm.ptr -> f32
    %173 = llvm.call @check_f32(%10, %157, %6, %2) : (i32, f32, f32, f32) -> i32
    %174 = llvm.call @check_f32(%9, %162, %5, %2) : (i32, f32, f32, f32) -> i32
    %175 = llvm.call @check_f32(%8, %167, %4, %2) : (i32, f32, f32, f32) -> i32
    %176 = llvm.call @check_f32(%7, %172, %3, %2) : (i32, f32, f32, f32) -> i32
    %177 = llvm.add %173, %174  : i32
    %178 = llvm.add %175, %176  : i32
    %179 = llvm.add %177, %178  : i32
    llvm.call @free(%77) : (!llvm.ptr) -> ()
    llvm.return %179 : i32
  }
}
