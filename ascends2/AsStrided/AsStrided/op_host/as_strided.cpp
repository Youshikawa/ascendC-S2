
#include "as_strided_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

/*
思路：
    以y的vec为单位来搬运
    首先计算当前vec首个元素在x中的pos 
    之后循环按stride最后一个遍历x
    在搬运x时应计算当前最多拿到的个数
*/


namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    AsStridedTilingData tiling;
    int32_t NUM = 4;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    auto shapeX = context->GetInputShape(0)->GetStorageShape();
    auto shapeY = context->GetOutputShape(0)->GetStorageShape();

    uint32_t yDims = shapeY.GetDimNum();    // y的dims as well as size and stride
    uint32_t yShape[64] = {0};
    uint32_t yShapeSum[64] = {0};
    // std::cout << "yDims = " << yDims << "\n";
    for (int i = 0; i < yDims; i ++) {
        yShape[i] = shapeY.GetDim(i);
    } 
    if (int32_t(yDims) - 2 >= 0)
        yShapeSum[yDims - 2] = 1;
    for (int i = int32_t(yDims) - 3; i >= 0; i --) {
        yShapeSum[i] = yShapeSum[i + 1] * yShape[i + 1];
    } 
    // for (int i = 0; i < yDims; i ++) {
    //     std::cout << yShape[i] << (i == yDims - 1 ? "\n" : " ");
    // }
    
    // for (int i = 0; i < yDims; i ++) {
    //     std::cout << yShapeSum[i] << (i == yDims - 1 ? "\n" : " ");
    // }
    uint32_t totalLengthStride = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    uint32_t totalLengthX = shapeX.GetShapeSize();
    uint32_t totalLengthY = shapeY.GetShapeSize();
     

    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        //NUM = 3;
    }else{
        //NUM = 9;
        sizeofdatatype = 4;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; 
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / BUFFER_NUM) / NUM;  
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = 1;

    uint32_t strideArr[64] = {0};
    tiling.set_totalLengthX(totalLengthX);
    tiling.set_totalLengthY(totalLengthY);
    tiling.set_totalLengthStride(totalLengthStride);
    tiling.set_yDims(yDims);
    tiling.set_yShape(yShape);
    tiling.set_yShapeSum(yShapeSum);
    tiling.set_strideArr(strideArr);
    tiling.set_vecCount(totalLengthY / yShape[yDims - 1]);

    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);  
    
    // printf("##########################################################################\n");
    // printf("                            THIS IS TILING DATAS\n");
    // std::cout << "++ " << "totalLengthX = " << totalLengthX << "\n";
    // std::cout << "++ " << "totalLengthY = " << totalLengthY << "\n";
    // std::cout << "++ " << "totalLengthStride = " << totalLengthStride << "\n";
    // std::cout << "++ " << "block_size = " << block_size << "\n";
    // std::cout << "++ " << "vecCount = " << totalLengthY / yShape[yDims - 1] << "\n"; 
    // std::cout << "++ " << "vecLength = " << yShape[yDims - 1] << "\n";
    // printf("##########################################################################\n");



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
class AsStrided : public OpDef {
public:
    explicit AsStrided(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("stride")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("storage_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(AsStrided);
}
