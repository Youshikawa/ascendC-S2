
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GlobalAvgPoolTilingData)
    TILING_DATA_FIELD_DEF(float, chanelLengthFloat); 

    TILING_DATA_FIELD_DEF(uint32_t, totalLengthY);
    TILING_DATA_FIELD_DEF(uint32_t, blockLengthY);
    
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthX);
    TILING_DATA_FIELD_DEF(uint32_t, blockLengthX);

    TILING_DATA_FIELD_DEF(uint32_t, chanelLength);
    TILING_DATA_FIELD_DEF(uint32_t, chanelLengthAligned);
    
    TILING_DATA_FIELD_DEF(uint32_t, chanelCount);
    TILING_DATA_FIELD_DEF(uint32_t, tileNumChanel);
    TILING_DATA_FIELD_DEF(uint32_t, finalLoopLengthChanel);
    TILING_DATA_FIELD_DEF(uint32_t, finalLoopCacLengthChanel);

    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM); 
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GlobalAvgPool, GlobalAvgPoolTilingData)
}
