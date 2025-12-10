/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ADORAONNX/Conversion/ONNXToADORACommon.hpp"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#define ONNX_ML
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#define DEBUG_TYPE "onnx-to-adora"

using namespace mlir;

namespace mlir {
namespace ADORA {
namespace ADORATensor {

/// KrnlGlobalOp -> memref.global + memref.get_global
///
/// krnl.global @name(...) : memref<...> = dense<...>
///   =>
/// memref.global @name : memref<...> = dense<...>
/// %0 = memref.get_global @name : memref<...>
class KrnlGlobalOpToMemRefGlobalPattern
    : public OpConversionPattern<KrnlGlobalOp> {
public:
  using OpConversionPattern<KrnlGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(KrnlGlobalOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override {
  Location loc = op.getLoc();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    return rewriter.notifyMatchFailure(
        op, "expected KrnlGlobalOp to be inside a ModuleOp");

  auto memRefTy = op.getResult().getType().dyn_cast<MemRefType>();
  if (!memRefTy)
    return rewriter.notifyMatchFailure(
        op, "KrnlGlobalOp result is not a MemRefType");

  // -----------------------------
  // 1. insert memref.global
  // -----------------------------
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  StringAttr nameAttr = op.getNameAttr();

  auto valueAttrOpt = op.getValue();      // std::optional<Attribute> / Optional<Attribute>
  Attribute initValue =
      valueAttrOpt ? *valueAttrOpt : Attribute();  

  bool isConstant = valueAttrOpt.has_value();

  IntegerAttr alignmentAttr = op.getAlignmentAttr();   // 可能为空

  auto visibilityAttr =
      op->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName());

  auto global = rewriter.create<memref::GlobalOp>(
      loc,
      /*sym_name=*/nameAttr,
      /*sym_visibility=*/visibilityAttr,
      /*type=*/memRefTy,
      /*initial_value=*/initValue,   
      /*constant=*/isConstant,
      /*alignment=*/alignmentAttr);

    // -----------------------------
    // 2. insert memref.get_global to replace KrnlGlobalOp
    // -----------------------------
    rewriter.setInsertionPoint(op);
    Value newMemRef =
        rewriter.create<memref::GetGlobalOp>(loc, memRefTy, nameAttr);
    rewriter.replaceOp(op, newMemRef);

    return success();
  }
};

void populateAdoraLoweringKrnlGlobalToMemRefGlobal(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<KrnlGlobalOpToMemRefGlobalPattern>(ctx);
}

} // namespace AdoraTensor
} // namespace Adora
} // namespace mlir
