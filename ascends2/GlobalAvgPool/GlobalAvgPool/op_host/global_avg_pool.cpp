
#include "global_avg_pool_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    GlobalAvgPoolTilingData tiling;
    int32_t NUM = 6;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    auto shapeX = context->GetInputShape(0)->GetStorageShape(); 
    uint32_t totalLengthX = shapeX.GetShapeSize();
    uint32_t chanelLength = 1;
    for (int i = shapeX.GetDimNum() - 1; i > 1; i --) {
        chanelLength *= shapeX.GetDim(i);
    }
    uint32_t chanelCount = totalLengthX / chanelLength;
    float chanelLengthFloat = float(chanelLength);
    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 12;
    }else{
        //NUM = 9;
        sizeofdatatype = 4;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; 
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / BUFFER_NUM) / NUM;  
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = 1;    
    uint32_t totalLengthY = chanelCount;
    uint32_t extraLengthY = totalLengthY % ALIGN_NUM ? ALIGN_NUM - totalLengthY % ALIGN_NUM : 0;
    uint32_t blockLengthY = totalLengthY + extraLengthY; // aligned

    uint32_t extraLengthX = totalLengthX % ALIGN_NUM ? ALIGN_NUM - totalLengthX % ALIGN_NUM : 0;
    uint32_t blockLengthX = totalLengthX + extraLengthX; // aligned

    uint32_t extraChanelLength = chanelLength  % ALIGN_NUM ? ALIGN_NUM - chanelLength % ALIGN_NUM : 0;
    uint32_t chanelLengthAligned = chanelLength + extraChanelLength; // aligned
    
    uint32_t tileNumChanel = chanelLengthAligned / block_size + (chanelLengthAligned % block_size > 0);
    uint32_t finalLoopLengthChanel = chanelLengthAligned - (tileNumChanel - 1) * block_size;
    uint32_t finalLoopCacLengthChanel = chanelLength - (tileNumChanel - 1) * block_size;


    tiling.set_totalLengthY(totalLengthY);
    tiling.set_blockLengthY(blockLengthY);

    tiling.set_totalLengthX(totalLengthX);
    tiling.set_blockLengthX(blockLengthX);

    tiling.set_chanelLength(chanelLength);
    tiling.set_chanelLengthAligned(chanelLengthAligned);

    tiling.set_chanelCount(chanelCount);
    tiling.set_chanelLengthFloat(1.0 / chanelLengthFloat); 
    
    tiling.set_tileNumChanel(tileNumChanel);
    tiling.set_finalLoopLengthChanel(finalLoopLengthChanel);
    tiling.set_finalLoopCacLengthChanel(finalLoopCacLengthChanel);

    tiling.set_ALIGN_NUM(ALIGN_NUM); 
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);  
    printf("##########################################################################\n");
    printf("                            THIS IS TILING DATAS\n");
    std::cout << "++ " << "totalLengthX = " << totalLengthX << "\n";
    std::cout << "++ " << "blockLengthX = " << blockLengthX << "\n";

    std::cout << "++ " << "totalLengthY = " << totalLengthY << "\n";
    std::cout << "++ " << "blockLengthY = " << blockLengthY << "\n";

    std::cout << "++ " << "chanelLength = " << chanelLength << "\n";
    std::cout << "++ " << "chanelLengthAligned = " << chanelLengthAligned << "\n";

    std::cout << "++ " << "chanelCount = " << chanelCount << "\n";

    std::cout << "++ " << "tileNumChanel = " << tileNumChanel << "\n";
    std::cout << "++ " << "finalLoopLengthChanel = " << finalLoopLengthChanel << "\n";
    std::cout << "++ " << "finalLoopCacLengthChanel = " << finalLoopCacLengthChanel << "\n";


    std::cout << "++ " << "block_size = " << block_size << "\n";  
    printf("##########################################################################\n");



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
class GlobalAvgPool : public OpDef {
public:
    explicit GlobalAvgPool(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GlobalAvgPool);
}
