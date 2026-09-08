; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_2xf32 = private constant [2 x float] [float -6.000000e+00, float 1.000000e+00], align 64
@__constant_2x2xf32_0 = private constant [2 x [2 x float]] [[2 x float] [float 1.000000e+00, float -1.000000e+00], [2 x float] [float 2.000000e+00, float 3.000000e+00]], align 64
@__constant_2x2xf32 = private constant [2 x [2 x float]] [[2 x float] [float 1.000000e+00, float 2.000000e+00], [2 x float] [float 3.000000e+00, float 4.000000e+00]], align 64

declare void @free(ptr)

declare ptr @malloc(i64)

declare i32 @check_f32(i32, float, float, float)

define i32 @main() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 4) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 2, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 2, 3, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 2, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 1, 4, 1
  br label %14

14:                                               ; preds = %26, %0
  %15 = phi i64 [ %27, %26 ], [ 0, %0 ]
  %16 = icmp slt i64 %15, 2
  br i1 %16, label %17, label %28

17:                                               ; preds = %14
  br label %18

18:                                               ; preds = %21, %17
  %19 = phi i64 [ %25, %21 ], [ 0, %17 ]
  %20 = icmp slt i64 %19, 2
  br i1 %20, label %21, label %26

21:                                               ; preds = %18
  %22 = mul i64 %15, 2
  %23 = add i64 %22, %19
  %24 = getelementptr float, ptr %6, i64 %23
  store float 0.000000e+00, ptr %24, align 4
  %25 = add i64 %19, 1
  br label %18

26:                                               ; preds = %18
  %27 = add i64 %15, 1
  br label %14

28:                                               ; preds = %14
  br label %29

29:                                               ; preds = %61, %28
  %30 = phi i64 [ %62, %61 ], [ 0, %28 ]
  %31 = icmp slt i64 %30, 2
  br i1 %31, label %32, label %63

32:                                               ; preds = %29
  br label %33

33:                                               ; preds = %59, %32
  %34 = phi i64 [ %60, %59 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 2
  br i1 %35, label %36, label %61

36:                                               ; preds = %33
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %58, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 2
  br i1 %39, label %40, label %59

40:                                               ; preds = %37
  %41 = mul i64 %30, 2
  %42 = add i64 %41, %38
  %43 = getelementptr float, ptr @__constant_2x2xf32, i64 %42
  %44 = load float, ptr %43, align 4
  %45 = mul i64 %38, 2
  %46 = add i64 %45, %34
  %47 = getelementptr float, ptr @__constant_2x2xf32_0, i64 %46
  %48 = load float, ptr %47, align 4
  %49 = mul i64 %30, 2
  %50 = add i64 %49, %34
  %51 = getelementptr float, ptr %6, i64 %50
  %52 = load float, ptr %51, align 4
  %53 = fmul float %44, %48
  %54 = fadd float %52, %53
  %55 = mul i64 %30, 2
  %56 = add i64 %55, %34
  %57 = getelementptr float, ptr %6, i64 %56
  store float %54, ptr %57, align 4
  %58 = add i64 %38, 1
  br label %37

59:                                               ; preds = %37
  %60 = add i64 %34, 1
  br label %33

61:                                               ; preds = %33
  %62 = add i64 %30, 1
  br label %29

63:                                               ; preds = %29
  br label %64

64:                                               ; preds = %84, %63
  %65 = phi i64 [ %85, %84 ], [ 0, %63 ]
  %66 = icmp slt i64 %65, 2
  br i1 %66, label %67, label %86

67:                                               ; preds = %64
  br label %68

68:                                               ; preds = %71, %67
  %69 = phi i64 [ %83, %71 ], [ 0, %67 ]
  %70 = icmp slt i64 %69, 2
  br i1 %70, label %71, label %84

71:                                               ; preds = %68
  %72 = getelementptr float, ptr @__constant_2xf32, i64 %69
  %73 = load float, ptr %72, align 4
  %74 = mul i64 %65, 2
  %75 = add i64 %74, %69
  %76 = getelementptr float, ptr %6, i64 %75
  %77 = load float, ptr %76, align 4
  %78 = fadd float %77, %73
  %79 = call float @llvm.maximum.f32(float %78, float 0.000000e+00)
  %80 = mul i64 %65, 2
  %81 = add i64 %80, %69
  %82 = getelementptr float, ptr %6, i64 %81
  store float %79, ptr %82, align 4
  %83 = add i64 %69, 1
  br label %68

84:                                               ; preds = %68
  %85 = add i64 %65, 1
  br label %64

86:                                               ; preds = %64
  %87 = getelementptr float, ptr %6, i64 0
  %88 = load float, ptr %87, align 4
  %89 = getelementptr float, ptr %6, i64 1
  %90 = load float, ptr %89, align 4
  %91 = getelementptr float, ptr %6, i64 2
  %92 = load float, ptr %91, align 4
  %93 = getelementptr float, ptr %6, i64 3
  %94 = load float, ptr %93, align 4
  %95 = call i32 @check_f32(i32 0, float %88, float 0.000000e+00, float 0x3EE4F8B580000000)
  %96 = call i32 @check_f32(i32 1, float %90, float 6.000000e+00, float 0x3EE4F8B580000000)
  %97 = call i32 @check_f32(i32 2, float %92, float 5.000000e+00, float 0x3EE4F8B580000000)
  %98 = call i32 @check_f32(i32 3, float %94, float 1.000000e+01, float 0x3EE4F8B580000000)
  %99 = add i32 %95, %96
  %100 = add i32 %97, %98
  %101 = add i32 %99, %100
  call void @free(ptr %1)
  ret i32 %101
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
