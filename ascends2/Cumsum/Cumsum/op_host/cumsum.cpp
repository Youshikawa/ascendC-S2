
#include "cumsum_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    CumsumTilingData tiling;
    int32_t NUM = 15;
    uint32_t sizeofdatatype; 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    auto exclusive = context->GetAttrs()->GetBool(0);
    auto reverse = context->GetAttrs()->GetBool(1); 
    auto shapeX = context->GetInputShape(0)->GetStorageShape();
    uint32_t dims[32];
    uint32_t dimSum[32];
    for (int i = 0; i < 32; i ++) {
        dims[i] = dimSum[i] = 1;
    }
    uint32_t dimNum = shapeX.GetDimNum();
    for (int i = 0; i < dimNum; i ++) {
        dims[i] = shapeX.GetDim(i);
    }
    for (int i = dimNum - 1; i >= 0; i --) {
        dimSum[i] = dimSum[i + 1];
        dimSum[i] *= dims[i + 1];
    }
    auto dt = context->GetInputDesc(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        // NUM = 20;
    }else{ 
        sizeofdatatype = 4;
        // NUM = 12;
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype; 
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;  
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = 1;
 
    uint32_t totalLength = shapeX.GetShapeSize(); 
    uint32_t blockLength = totalLength + (totalLength % ALIGN_NUM ? ALIGN_NUM - totalLength % ALIGN_NUM : 0);

    tiling.set_totalLength(totalLength); 
    tiling.set_blockLength(blockLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM); 
    tiling.set_block_size(block_size); 
    tiling.set_exclusive(*exclusive);
    tiling.set_reverse(*reverse);   
    tiling.set_dimNum(dimNum);
    tiling.set_dims(dims);
    tiling.set_dimSum(dimSum);
    context->SetBlockDim(aivNum);
    printf("##########################################################################\n");
    printf("                            THIS IS TILING DATAS\n"); 
    std::cout << "++ " << "totalLength = " << totalLength << "\n";
    std::cout << "++ " << "blockLength = " << blockLength << "\n";
    std::cout << "++ " << "ALIGN_NUM = " << ALIGN_NUM << "\n";  
    std::cout << "++ " << "block_size = " << block_size << "\n"; 
    std::cout << "++ " << "reverse = " << *reverse << "\n"; 
    std::cout << "++ " << "exclusive = " << *exclusive << "\n"; 

    std::cout << "++ " << "dimNum = " << dimNum << "\n"; 
     for (int i = 0; i < dimNum; i ++) {
        std::cout << dims[i] << " \n"[i == dimNum - 1];
    }

     for (int i = 0; i < dimNum; i ++) {
        std::cout << dimSum[i] << " \n"[i == dimNum - 1];
    }
    printf("##########################################################################\n");
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
class Cumsum : public OpDef {
public:
    explicit Cumsum(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("axis")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("exclusive").AttrType(OPTIONAL).Bool(false);
        this->Attr("reverse").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Cumsum);
}
