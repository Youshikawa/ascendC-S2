#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template<typename TYPE_X1,typename TYPE_Y> class Triu {
    using T = TYPE_X1;
public:
    __aicore__ inline Triu() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, uint32_t matSize, uint32_t matCount, int32_t diagonal, uint32_t vecLength, uint32_t vecNum,  uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0); // 114688
        this->isLastAiCore = GetBlockNum() == GetBlockIdx() + 1; //true
        this->tileLength = block_size;  //9024
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); // 11472
        this->totalLength = totalLength;  
        this->ALIGN_NUM = ALIGN_NUM;  //8
        this->vecNum = vecNum;  
        this->vecLength = vecLength;  
        this->diagonal = diagonal; // 0
        this->blockIdx = GetBlockIdx(); // 0
        this->matCount = matCount;
        this->matSize = matSize;

        auto startPointer = core_size * GetBlockIdx(); //0
        auto bufferlength = this->blockLength; //11472

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferlength); //11472
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));  
        pipe.InitBuffer(B_zeros, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(B_ones, this->tileLength * sizeof(uint8_t));
        //pipe.InitBuffer(B_bzeros, this->tileLength * sizeof(uint8_t));
        this->zeros = B_zeros.Get<TYPE_X1>(); 
        this->ones = B_ones.Get<uint8_t>();
        //this->bzeros = B_zeros.Get<uint8_t>();
        Duplicate(this->zeros, TYPE_X1(0), this->tileLength);
        Compare(this->ones, this->zeros, this->zeros, CMPMODE::EQ, this->tileLength);
       // Compare(this->bzeros, this->zeros, this->zeros, CMPMODE::NE, this->tileLength);
        //Duplicate<uint16_t>(this->ones, uint16_t(1), this->tileLength);
    }
    __aicore__ inline void Process() {
        int32_t count = this->vecNum; // 224
        uint32_t totalLength = this->vecLength; // 1096
        uint32_t matLength = this->matSize;     
        for (int64_t k = 0; k < this->matCount; k ++) {
            for (int64_t j = 0; j < count; j ++) { // 224
                int32_t loopCount = totalLength / this->tileLength + (totalLength % this->tileLength > 0); // 1
                for (int32_t i = 0; i < loopCount - 1; i ++) { 
                    CopyIn(k * matLength + j * totalLength, i, this->tileLength);
                    Compute(i, j + this->diagonal - (i * this->tileLength), this->tileLength);
                    CopyOut(k * matLength + j * totalLength, i, this->tileLength);
                }
                uint32_t length = totalLength - this->tileLength * (loopCount - 1); // 1096
                auto length_align = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM; // 1096
                CopyIn(k * matLength + j * totalLength, loopCount - 1, length_align); // 0, 0, 
                Compute(loopCount - 1, j + this->diagonal - ((loopCount - 1) * this->tileLength), length); // 0, 0, 1321
                CopyOut(k * matLength + j * totalLength, loopCount - 1, length_align); // 0, 0, 1344
            } 
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t start, int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        DataCopy(x1, Gm_x1[start + progress * this->tileLength], length); // x1 copy了第一个vec
        Q_x1.EnQue(x1);
    }
    // 小于 pos的都为0 大于等于 pos的留下
    __aicore__ inline void Compute(int32_t progress, int32_t pos, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        //Select(y, ones, zeros, x1, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
         if (pos <= 0) { // pos <= 0 == 切片首地址 -> 切片首地址大于等于pos
             //Select(y, ones, zeros, x1, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
            Select(y, this->ones, x1, TYPE_X1(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            //Add(y, x1, zeros, length);
         } else if (pos >= length) { // pos >= length > length - 1 = 切片尾地址 全为0
             //Select(y, ones, x1, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, length);
            //Add(y, zeros, zeros, length);
            Select(y, this->ones, zeros, TYPE_X1(-2), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
         } else {
            //  Select(y, ones, zeros, x1, SELMODE::VSEL_TENSOR_TENSOR_MODE, length); // 先全选X
            //  Select(y, ones, x1, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, pos - 1); // 余下重新补0
            // Add(y, x1, zeros, length);
            Select(y, this->ones, x1, TYPE_X1(-3), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            // Add(y, zeros, zeros, pos);
            Select(y, this->ones, zeros, TYPE_X1(-4), SELMODE::VSEL_TENSOR_SCALAR_MODE, pos);
         }
        Q_x1.FreeTensor(x1); 
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(uint32_t start, int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[start + progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1; // X
    TBuf<QuePosition::VECCALC> B_zeros, B_ones; // X X X u16
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y; // X
    GlobalTensor<TYPE_X1> Gm_x1;      // 5 * X + 16
    GlobalTensor<TYPE_Y> Gm_y;
    LocalTensor<TYPE_X1> zeros; 
    LocalTensor<uint8_t> ones; 
    
    bool isLastAiCore = false;
    uint32_t blockIdx;
    int32_t diagonal;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t vecNum;
    uint32_t totalLength;
    uint32_t ALIGN_NUM;
    uint32_t vecLength;
    uint32_t matCount;
    uint32_t matSize;
};


extern "C" __global__ __aicore__ void triu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    Triu<DTYPE_X, DTYPE_Y> op; 
    op.Init(x, y, tiling_data.matSize, tiling_data.matCount, tiling_data.diagonal, tiling_data.vecLength, tiling_data.vecNum, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}