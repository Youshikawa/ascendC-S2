
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HistogramTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthX); 
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthY); 
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum); 
    TILING_DATA_FIELD_DEF(int32_t, bins);
    TILING_DATA_FIELD_DEF(float, min);
    TILING_DATA_FIELD_DEF(float, max);
    TILING_DATA_FIELD_DEF(float, area);
    TILING_DATA_FIELD_DEF(bool, cacMinMax);
    TILING_DATA_FIELD_DEF(float, recBins);
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Histogram, HistogramTilingData)
}
