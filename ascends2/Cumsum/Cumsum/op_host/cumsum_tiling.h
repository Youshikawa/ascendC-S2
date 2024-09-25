
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CumsumTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);  
    TILING_DATA_FIELD_DEF(uint32_t, ALIGN_NUM);
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);

    // TILING_DATA_FIELD_DEF(uint32_t, totalCumsumCount); // 总共要对多少个tensor进行内部累加 即 总的除一下
    // TILING_DATA_FIELD_DEF(uint32_t, oneCumsumLength); // 一次内部累加要处理多少元素 
    // TILING_DATA_FIELD_DEF(uint32_t, currentAixsCount); // 一次完整的累加所涉及的次数 比如 最后一维就是1  b y x 第二维就是 y 的维度
    // TILING_DATA_FIELD_DEF(uint32_t, aixsOffsetLength); // 累加与上一个tensor的偏移量，即aixs + 1的的shapesize 
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, dimSum);
    TILING_DATA_FIELD_DEF(uint32_t, dimNum);

    TILING_DATA_FIELD_DEF(bool, exclusive);
    TILING_DATA_FIELD_DEF(bool, reverse);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cumsum, CumsumTilingData)
}
