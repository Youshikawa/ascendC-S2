
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceSumTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);   
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthAxes);
    TILING_DATA_FIELD_DEF(uint32_t, blockLengthAxes);
    TILING_DATA_FIELD_DEF(uint32_t, blockLengthY);
       
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, dimSum);
    TILING_DATA_FIELD_DEF(uint32_t, dimNum);

    TILING_DATA_FIELD_DEF(bool, ignore_nan); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceSum, ReduceSumTilingData)
}
