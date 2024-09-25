
#include "three_nn_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <assert.h>

/*一个一个算太慢了 必须用mask */
namespace optiling {
    const int32_t BLOCK_SIZE = 32;
    const int32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ThreeNNTilingData tiling;
    int32_t NUM = 9; 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto shape_bn3 = context->GetInputShape(0)->GetStorageShape(); 
    auto shape_bm3 = context->GetInputShape(1)->GetStorageShape();

    uint32_t batchCount;
    uint32_t nCount;

    nCount = shape_bn3.GetDim(1); // 0 b   1 n   2 3
    batchCount = shape_bn3.GetDim(0);

    uint32_t ALIGN_NUM = BLOCK_SIZE / 4; // 8 
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / BUFFER_NUM) / NUM;   
    uint32_t block_size = 64; // 这里一定是 3的倍数 且是256字节对其的
    /* 如果 run failed了 很可能是 block_size 为 0 了 */
    if (block_size % 3 || block_size == 0) {
        assert("TILING WRONG!!!!!!!!!!");
    }  
    uint32_t tilePointCount = block_size / 3; // 
    uint32_t aivNum = 1;    
    uint32_t totalLength_bn3 = shape_bn3.GetShapeSize();
    uint32_t batchLength_bn3 = totalLength_bn3 / batchCount;  
    uint32_t extraLength_bn3 = totalLength_bn3 % ALIGN_NUM ? ALIGN_NUM - totalLength_bn3 % ALIGN_NUM : 0;
    uint32_t blockLength_bn3 = totalLength_bn3 + extraLength_bn3; // aligned

    uint32_t totalLength_bm3 = shape_bm3.GetShapeSize();
    uint32_t batchLength_bm3 = totalLength_bm3 / batchCount;
    uint32_t extraLength_bm3 = totalLength_bm3 % ALIGN_NUM ? ALIGN_NUM - totalLength_bm3 % ALIGN_NUM : 0;
    uint32_t blockLength_bm3 = totalLength_bm3 + extraLength_bm3; // aligned
 
    uint32_t loopCount = batchLength_bm3 / 63 + (batchLength_bm3 % 63 > 0);
    uint32_t finalLoopLength = batchLength_bm3 - (loopCount - 1) * 63;  // aligned 用于搬运
    uint32_t finalLoopCacLength = finalLoopLength;
    finalLoopLength = finalLoopLength + (finalLoopLength % ALIGN_NUM ? ALIGN_NUM - finalLoopLength % ALIGN_NUM : 0);
    uint64_t tileMask[2] = {0,0};
    for (int32_t i = 0; i < 21; i ++) {
        tileMask[0] <<= 3;
        tileMask[0] |= 4;
    }
    tileMask[0] <<= 1;
    


    if (finalLoopCacLength % 3) {
        assert("TILING WRONG!!!!!!!!!!");
    }
    uint32_t finalLoopPointCount = finalLoopCacLength / 3;
    uint64_t finalMask[2] = {0,0};
    uint32_t offset = 64;
    for (int32_t i = 0; i < finalLoopPointCount; i ++) {
        finalMask[0] |= (1 << offset);
        offset -= 3;
    }
    
    tiling.set_tilePointCount(tilePointCount);
    tiling.set_finalLoopPointCount(finalLoopPointCount);

    tiling.set_totalLength_bn3(totalLength_bn3);
    tiling.set_blockLength_bn3(blockLength_bn3);

    tiling.set_totalLength_bm3(totalLength_bm3);
    tiling.set_blockLength_bm3(blockLength_bm3);
    
    tiling.set_batchLength_bn3(batchLength_bn3);
    tiling.set_batchLength_bm3(batchLength_bm3);
    
    tiling.set_batchCount(batchCount); 
    tiling.set_nCount(nCount);
    
    tiling.set_loopCount(loopCount);
    tiling.set_finalLoopLength(finalLoopLength);
    tiling.set_finalLoopCacLength(finalLoopCacLength);

    tiling.set_ALIGN_NUM(ALIGN_NUM); 
    tiling.set_block_size(block_size);
    // tiling.set_aivNum(aivNum);  
    // printf("##########################################################################\n");
    // printf("                            THIS IS TILING DATAS\n");
    // std::cout << "++ " << "totalLength_bn3 = " << totalLength_bn3 << "\n";
    // std::cout << "++ " << "blockLength_bn3 = " << blockLength_bn3 << "\n";

    // std::cout << "++ " << "totalLength_bm3 = " << totalLength_bm3 << "\n";
    // std::cout << "++ " << "blockLength_bm3 = " << blockLength_bm3 << "\n";

    // std::cout << "++ " << "batchLength_bn3 = " << batchLength_bn3 << "\n";
    // std::cout << "++ " << "batchLength_bm3 = " << batchLength_bm3 << "\n";

    // std::cout << "++ " << "batchCount = " << batchCount << "\n";
    // std::cout << "++ " << "nCount = " << nCount << "\n";
    
    // std::cout << "++ " << "loopCount = " << loopCount << "\n";
    // std::cout << "++ " << "finalLoopLength = " << finalLoopLength << "\n";
    // std::cout << "++ " << "finalLoopCacLength = " << finalLoopCacLength << "\n";

    // std::cout << "++ " << "tilePointCount = " << tilePointCount << "\n";
    // std::cout << "++ " << "finalLoopPointCount = " << finalLoopPointCount << "\n";


    // std::cout << "++ " << "block_size = " << block_size << "\n";  
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
class ThreeNN : public OpDef {
public:
    explicit ThreeNN(const char* name) : OpDef(name)
    {
        this->Input("xyz1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("xyz2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ThreeNN);
}
