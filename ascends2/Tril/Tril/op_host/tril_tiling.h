
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TrilTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
  TILING_DATA_FIELD_DEF(uint32_t, tiling_size);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, aivNum);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);
  TILING_DATA_FIELD_DEF(int32_t, diagonal);
  TILING_DATA_FIELD_DEF(uint32_t, vecLength);
  TILING_DATA_FIELD_DEF(uint32_t, matCount);
  TILING_DATA_FIELD_DEF(uint32_t, matSize);
  TILING_DATA_FIELD_DEF(uint32_t, vecNum);  // 1 1 1 1 1
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Tril, TrilTilingData)
}
