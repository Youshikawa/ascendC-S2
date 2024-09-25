
#include "xlogy_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  XlogyTilingData tiling;
  int32_t NUM = 15;    // 唯一需要根据程序作出调整
    uint32_t sizeofdatatype; 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum(); 
    /*
        获取shape,attr等初始资源
    */
    auto shapeX1 = context->GetInputShape(0)->GetStorageShape();
    auto shapeX2 = context->GetInputShape(1)->GetStorageShape();
    auto shapeY = context->GetOutputShape(0)->GetStorageShape();
    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
        NUM = 25;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 8;
    }else{ 
        sizeofdatatype = 4;
        NUM = 8;
    }
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; 
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / BUFFER_NUM) / NUM;  
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint32_t block_size = tiling_size * ALIGN_NUM;
    uint32_t totalLengthX1 = shapeX1.GetShapeSize(); 
    uint32_t blockLengthX1 = totalLengthX1 + (totalLengthX1 % ALIGN_NUM ? ALIGN_NUM - totalLengthX1 % ALIGN_NUM : 0);
    uint32_t totalLengthX2 = shapeX2.GetShapeSize(); 
    uint32_t blockLengthX2 = totalLengthX2 + (totalLengthX2 % ALIGN_NUM ? ALIGN_NUM - totalLengthX2 % ALIGN_NUM : 0);
    
    uint32_t totalLengthY = shapeY.GetShapeSize();
    uint32_t blockLengthY = totalLengthY + (totalLengthY % ALIGN_NUM ? ALIGN_NUM - totalLengthY % ALIGN_NUM : 0);   


    tiling.set_totalLengthX1(totalLengthX1);
    tiling.set_blockLengthX1(blockLengthX1);
    tiling.set_totalLengthX1(totalLengthX2);
    tiling.set_blockLengthX1(blockLengthX2);
    tiling.set_totalLengthY(totalLengthY);
    tiling.set_blockLengthY(blockLengthY);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_block_size(block_size);
    printf("totalLength %d %d %d %d %d %d %d\n",totalLengthX1,blockLengthX2,totalLengthX2,blockLengthX1,totalLengthY,blockLengthY,block_size);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Xlogy : public OpDef {
public:
    explicit Xlogy(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Xlogy);
}
