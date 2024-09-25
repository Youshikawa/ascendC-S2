
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ThreeNNTilingData) 

    TILING_DATA_FIELD_DEF(uint32_t, totalLength_bn3);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength_bn3);
    
    TILING_DATA_FIELD_DEF(uint32_t, totalLength_bm3);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength_bm3);
   
    TILING_DATA_FIELD_DEF(uint32_t, batchLength_bn3);
    TILING_DATA_FIELD_DEF(uint32_t, batchLength_bm3);

    TILING_DATA_FIELD_DEF(uint32_t, batchCount);
    TILING_DATA_FIELD_DEF(uint32_t, nCount);
    
    TILING_DATA_FIELD_DEF(uint32_t, loopCount);
    TILING_DATA_FIELD_DEF(uint32_t, finalLoopLength);
    TILING_DATA_FIELD_DEF(uint32_t, finalLoopCacLength);

    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM); 
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);  

    TILING_DATA_FIELD_DEF(uint32_t, tilePointCount);
    TILING_DATA_FIELD_DEF(uint32_t, finalLoopPointCount);

    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, tileMask);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2, finalMask);
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ThreeNN, ThreeNNTilingData)
}
