//===---------------- AdoraONNXTransposeLowering.cpp - Lowering Transpose Op ---------------===//
//
// This file lowers the ONNX Transpose Operator to Affine loops:
//   affine.for + affine.load + affine.store,
// instead of using krnl.memcpy.
//
//===----------------------------------------------------------------------===//

// #include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "ADORAONNX/Conversion/ONNXToADORACommon.hpp"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define ONNX_ML
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#define DEBUG_TYPE "onnx-to-adora"

using namespace onnx_mlir;

namespace mlir {
namespace ADORA {
namespace ADORATensor {
struct AdoraONNXTransposeOpLowerToAffine : public OpConversionPattern<ONNXTransposeOp> {
  using MDBuilder = MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
      MemRefBuilder, MathBuilder>;
  bool enableParallel = false;

  AdoraONNXTransposeOpLowerToAffine(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {
  }

  LogicalResult matchAndRewrite(ONNXTransposeOp transposeOp,
      ONNXTransposeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = transposeOp.getOperation();
    Location loc = ONNXLoc<ONNXTransposeOp>(op);
    ValueRange operands = adaptor.getOperands();
    MDBuilder create(rewriter, loc);

    // Operands and attributes.
    Value data = adaptor.getData();
    auto permAttr = adaptor.getPerm();

    // Input and output types.
    MemRefType inMemRefType = cast<MemRefType>(data.getType());
    Type outConvertedType =
        typeConverter->convertType(*op->result_type_begin());
    assert(outConvertedType && isa<MemRefType>(outConvertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outMemRefType = cast<MemRefType>(outConvertedType);

    // Compute output shape.
    ONNXTransposeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr &outDims = shapeHelper.getOutputDims();

    // If transpose does not permute non-1 dimensions, lower to a view (no data
    // movement).
    if (canBeViewOp(inMemRefType, permAttr)) {
      Value view = create.mem.reinterpretCast(data, outDims);
      rewriter.replaceOp(op, view);
      return success();
    }

    // Allocate the result buffer.
    Value alloc = create.mem.alignedAlloc(outMemRefType, outDims);

    // Lower transpose to explicit affine loop nest with affine.load/store.
    if (failed(scalarTransposeToAffine(
            op, data, alloc, permAttr, rewriter, loc))) {
      return op->emitError("AdoraONNXTransposeOpLowerToAffine to affine failed");
    }

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  // If transpose does not permute the non-1 dimensions, it is safe to lower
  // transpose to a view op.
  bool canBeViewOp(
      MemRefType inMemRefType, std::optional<ArrayAttr> permAttr) const {
    ArrayRef<int64_t> dims = inMemRefType.getShape();
    uint64_t rank = inMemRefType.getRank();

    SmallVector<int64_t, 4> originalAxes;
    for (uint64_t axis = 0; axis < dims.size(); ++axis)
      if (dims[axis] != 1)
        originalAxes.emplace_back(axis);

    SmallVector<int64_t, 4> permutedAxes;
    for (uint64_t i = 0; i < rank; ++i) {
      int64_t axis = ArrayAttrIntVal(permAttr, i);
      if (dims[axis] != 1)
        permutedAxes.emplace_back(axis);
    }
    return (originalAxes == permutedAxes);
  }

  /// Element-wise transpose lowered to affine.for + affine.load/store.
  ///
  /// Semantics:
  ///   for i0 in [0..N0)
  ///     for i1 in [0..N1)
  ///       ...
  ///         v = input[i0, i1, ..., i_{rank-1}]
  ///         output[ i_{perm[0]}, i_{perm[1]}, ..., i_{perm[rank-1]} ] = v
  ///
  LogicalResult scalarTransposeToAffine(Operation *op, Value inputMemRef,
      Value outputMemRef, std::optional<ArrayAttr> permAttr,
      PatternRewriter &rewriter, Location loc) const {
    auto inType = cast<MemRefType>(inputMemRef.getType());
    auto outType = cast<MemRefType>(outputMemRef.getType());
    unsigned rank = inType.getRank();

    if (rank == 0)
      return success();

    // For simplicity, require static shapes. This can be extended to dynamic
    // shapes using memref.dim + affine maps if needed.
    ArrayRef<int64_t> inShape = inType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();
    for (int64_t d : inShape)
      if (d == ShapedType::kDynamic)
        return op->emitError(
            "scalarTransposeToAffine currently expects static input shape");
    for (int64_t d : outShape)
      if (d == ShapedType::kDynamic)
        return op->emitError(
            "scalarTransposeToAffine currently expects static output shape");

    // Materialize permutation as an unsigned vector.
    SmallVector<unsigned, 4> perm(rank);
    for (unsigned i = 0; i < rank; ++i)
      perm[i] = static_cast<unsigned>(ArrayAttrIntVal(permAttr, i));

    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Value, 4> ivs; // Current loop induction variables.

    // Recursively build a perfectly nested affine.for loop nest.
    std::function<void(unsigned)> buildLoopNest = [&](unsigned dim) {
      if (dim == rank) {
        // Innermost loop body: perform one load and one store.
        SmallVector<Value, 4> loadIndices(ivs.begin(), ivs.end());

        // Output indices are permuted: out[i] = in[perm[i]].
        SmallVector<Value, 4> storeIndices(rank);
        for (unsigned i = 0; i < rank; ++i)
          storeIndices[i] = ivs[perm[i]];

        auto val = rewriter.create<affine::AffineLoadOp>(
            loc, inputMemRef, loadIndices);
        rewriter.create<affine::AffineStoreOp>(
            loc, val, outputMemRef, storeIndices);
        return;
      }

      int64_t ub = inShape[dim];
      auto forOp =
          rewriter.create<affine::AffineForOp>(loc, /*lb=*/0, /*ub=*/ub);
      ivs.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
      buildLoopNest(dim + 1);
      ivs.pop_back();
      // Insertion point is restored by OpBuilder::InsertionGuard.
    };

    buildLoopNest(/*dim=*/0);
    return success();
  }
};

void populateAdoraLoweringONNXTransposeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<AdoraONNXTransposeOpLowerToAffine>(typeConverter, ctx);
}

}
} // namespace
} // namespace