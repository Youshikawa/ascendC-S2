#include "tril_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

 

uint32_t NUM = 7;
const uint32_t BLOCK_SIZE = 32;
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TrilTilingData tiling;
    uint32_t sizeofdatatype; 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    //std::cout<<aivNum << "\n";
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto shape = context->GetInputShape(0)->GetStorageShape();
    int32_t dimNum = shape.GetDimNum();
    uint32_t totalVec = shape.GetDim(dimNum - 2);
    uint32_t vecLength = shape.GetDim(dimNum - 1);
    uint32_t matSize = totalVec * vecLength;
    uint32_t matCount = shape.GetShapeSize() / matSize;
    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
    }
    else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
    }
    else{
        sizeofdatatype = 4;
    }
 
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; 
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 1) / NUM; 
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8; 
 
    uint32_t block_size = tiling_size * ALIGN_NUM; 
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size); 
    //std::cout<<aivNum << "\n"; 
    aivNum = aivNum >= 1 ? aivNum : 1;                                                 
    //std::cout<<aivNum << "\n";
    aivNum = 1;                  
    
    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);  
    uint32_t core_remain = totalLength - aivNum * core_size;     
    // uint32_t vecNum[64] = {0};
    // uint32_t vecNumSum[65] = {0}; 
    //uint32_t offsetTailVec[aivNum] = {0};
 
    auto diagonal = context->GetAttrs()->GetInt(0);
    int64_t maxVal = std::max(totalVec, vecLength);
    int32_t diagonal_in = int32_t(*diagonal);
    if ((*diagonal) > maxVal) {
        diagonal_in = maxVal + 1;
    }
    if ((*diagonal) < 0) {
        if ((*diagonal) < -maxVal) {
            diagonal_in = -maxVal - 1;
        }
    }
    tiling.set_vecLength(vecLength);
    tiling.set_vecNum(totalVec);
    tiling.set_diagonal(diagonal_in);
    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    tiling.set_matCount(matCount);
    tiling.set_matSize(matSize);

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    // printf("##########################################################################\n");
    // printf("                            THIS IS TILING DATAS\n");
    // std::cout << "++ " << "ub_size = " << ub_size << "\n"; 
    // std::cout << "++ " << "totalLength = " << totalLength << "\n";
    // std::cout << "++ " << "ALIGN_NUM = " << ALIGN_NUM << "\n";
    // std::cout << "++ " << "tiling_size = " << tiling_size << "\n";
    // std::cout << "++ " << "block_size = " << block_size << "\n";
    // std::cout << "++ " << "aivNum = " << aivNum << "\n";
    // std::cout << "++ " << "core_size = " << core_size << "\n";
    // std::cout << "++ " << "core_remain = " << core_remain << "\n";
    // std::cout << "++ " << "vecLength = " << vecLength << "\n";
    // std::cout << "++ " << "totalVec = " << totalVec << "\n";
 
    // printf("##########################################################################\n");


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
class Tril : public OpDef {
public:
    explicit Tril(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("diagonal").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Tril);
}
