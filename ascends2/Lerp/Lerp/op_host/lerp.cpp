
#include "lerp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
//#include <iostream>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{   
    int BUFFER_NUM = 2;

    LerpTilingData tiling;
    int32_t NUM = 10;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    std::cout << aivNum << "\n";

    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 15;
        //printf("已经更新\n");
    }else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
        NUM = 10;
    }else{ //DT_FLOAT
        sizeofdatatype = 4;
        NUM = 10;
        //printf("已经更新\n");
    }// 固定



    auto inshape1 = context->GetInputShape(0)->GetStorageShape();
    auto inshape2 = context->GetInputShape(1)->GetStorageShape();
    auto inshape3 = context->GetInputShape(2)->GetStorageShape();
    auto outshape = context->GetOutputShape(0)->GetStorageShape();
 
    bool flag = false;

    for(int i=0;i<outshape.GetDimNum();i++){
        if(inshape1.GetDim(i) != inshape2.GetDim(i) || inshape1.GetDim(i) != inshape3.GetDim(i) 
            || inshape2.GetDim(i) != inshape3.GetDim(i)) {flag = true;  break;}   //如果有一个不一样就要广播
    }
     
    if(flag){
        BUFFER_NUM = 1;
        
        context->SetTilingKey(2);
        //printf("FLAGED !!!!!!!!!!!\n");
        auto outshapeDims = outshape.GetDimNum();
        std::vector<uint64_t> inshapeVec1(outshapeDims, 1),inshapeVec2(outshapeDims, 1),inshapeVec3(outshapeDims, 1);
        int idx = outshapeDims;
        for (int j = inshape1.GetDimNum() - 1; j >= 0; j --) {
            inshapeVec1[-- idx] = inshape1.GetDim(j);
        }
 
        idx = outshapeDims;
        for (int j = inshape2.GetDimNum() - 1; j >= 0; j --) {
            inshapeVec2[-- idx] = inshape2.GetDim(j);
        }// inshapeVec对齐，外围补1即可
        idx = outshapeDims;
        for (int j = inshape3.GetDimNum() - 1; j >= 0; j --) {
            inshapeVec3[-- idx] = inshape3.GetDim(j);
        }


        // std::cout << "START:\n";
        // for (int i = 0; i < outshapeDims; i ++) {
        //     std::cout << inshapeVec1[i] << " ";
        // }
        // std::cout << "\n";
        // std::cout << "END:\n";
        // for (int i = 0; i < outshapeDims; i ++) {
        //     std::cout << inshapeVec2[i] << " ";
        // }
        // std::cout << "\n";
        // std::cout << "WEIGHT:\n";
        // for (int i = 0; i < outshapeDims; i ++) {
        //     std::cout << inshapeVec3[i] << " ";
        // }
        // std::cout << "\n";


        uint32_t reduce1[20] = {0};
        uint32_t reduce2[20] = {0};
        uint32_t reduce3[20] = {0};
        uint32_t shape[20] = {0};
        uint32_t d = 1;
        for(int i = 0;i < outshape.GetDimNum();i ++) {
            shape[i] = outshape.GetDim(i);
            d *= outshape.GetDim(i);
            if(inshapeVec1[i] != outshape.GetDim(i)) reduce1[i] = 1; //outshape一定是最后的 ，reduce来测试哪一个维度需要改变
            if(inshapeVec2[i] != outshape.GetDim(i)) reduce2[i] = 1; 
            if(inshapeVec3[i] != outshape.GetDim(i)) reduce3[i] = 1; 
        }

        // std::cout << "START:\n";
        // for (int i = 0; i < outshapeDims; i ++) {
        //     std::cout << reduce1[i] << " ";
        // }
        // std::cout << "\n";
        // std::cout << "END:\n";
        // for (int i = 0; i < outshapeDims; i ++) {
        //     std::cout << reduce2[i] << " ";
        // }
        // std::cout << "\n";
        // std::cout << "WEIGHT:\n";
        // for (int i = 0; i < outshapeDims; i ++) {
        //     std::cout << reduce3[i] << " ";
        // }
        // std::cout << "\n";


        uint32_t dim = outshape.GetDimNum();
        for(int i=dim-1; i>=1; i--){
            if(!reduce1[i - 1] && !reduce2[i - 1] && !reduce3[i - 1] && !reduce1[i] && !reduce2[i] && !reduce3[i]){ //如果前后都一致才能降低一维 并把shape堆上去换成一维便于计算
                dim --;
                shape[i - 1] *= shape[i];
            }else{
                break;
            }
        }
        if(reduce1[dim - 1] || reduce2[dim - 1] || reduce3[dim - 1]) { //
            shape[dim] = 1;
            dim ++;
        }
        
        // std::cout << "SHAPE:\n";
        // for (int i = 0; i < dim; i ++) {
        //     std::cout << shape[i] << " ";
        // }
        // std::cout << "\n";

        tiling.set_shape(shape);
        tiling.set_reduce1(reduce1);
        tiling.set_reduce2(reduce2);
        tiling.set_reduce3(reduce3);
        tiling.set_dim(dim);
        aivNum = 1;
        totalLength = d;
        // printf("OVER\n");
    }else{
        context->SetTilingKey(1);
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 1) / NUM;
    uint32_t origin_tiling_size = tiling_size; //要删除 //440
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;  // 一次运算datablock个数，尽量满足8的倍数

    uint32_t block_size = tiling_size * ALIGN_NUM; // 一次运算能搬运元素的个数 因为乘了ALIGN_NUM所以一定是32字节对齐
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size); 
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8); // 一个核中的元素数量
    uint32_t core_remain = totalLength - aivNum * core_size; //最后一个核要补充的元素数量 
    // printf("##########################################################################\n");
    // printf("                            THIS IS TILING DATAS\n");
    // std::cout << "++ " << "ub_size = " << ub_size << "\n"; 
    // std::cout << "++ " << "totalLength = " << totalLength << "\n";
    // std::cout << "++ " << "ALIGN_NUM = " << ALIGN_NUM << "\n";
    // std::cout << "++ " << "origin_tiling_size = " << origin_tiling_size << "\n";
    // std::cout << "++ " << "tiling_size = " << tiling_size << "\n";
    // std::cout << "++ " << "block_size = " << block_size << "\n";
    // std::cout << "++ " << "aivNum = " << aivNum << "\n";
    // std::cout << "++ " << "core_size = " << core_size << "\n";
    // std::cout << "++ " << "core_remain = " << core_remain << "\n";
    // printf("##########################################################################\n");

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {

    template<class T>
    T Max(const T& a, const T& b) {
        return a > b ? a : b;
    }


    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}


namespace ops {
class Lerp : public OpDef {
public:
    explicit Lerp(const char* name) : OpDef(name)
    {
        this->Input("start")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("end")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
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
OP_ADD(Lerp); 
}