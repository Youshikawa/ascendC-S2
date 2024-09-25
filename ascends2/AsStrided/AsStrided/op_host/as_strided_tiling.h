
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsStridedTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthX);
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthY);
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthStride);
    TILING_DATA_FIELD_DEF(uint32_t, yDims); 
    TILING_DATA_FIELD_DEF(uint32_t, vecCount);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, yShape);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, yShapeSum);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, strideArr);

    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);  

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsStrided, AsStridedTilingData)
}
