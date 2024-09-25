
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(XlogyTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLengthX1);
  TILING_DATA_FIELD_DEF(uint32_t, totalLengthX2);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthX1); 
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthX2); 
  TILING_DATA_FIELD_DEF(uint32_t, totalLengthY);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthY);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Xlogy, XlogyTilingData)
}
