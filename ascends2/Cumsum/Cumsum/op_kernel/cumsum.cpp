#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
/*

不能用 tmp 记录 因为 有可能记录到上一次 tile
reverse 应该也可通过上个解决
其他计算逻辑没有问题
*/

template<typename TYPE_X1, typename TYPE_Y> class Cumsum {
    using T = TYPE_X1;
public:
    __aicore__ inline Cumsum() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, bool exclusive, bool reverse, uint32_t* dims, uint32_t* dimSum, uint32_t dimNum,
                                 uint32_t ALIGN_NUM, uint32_t block_size, uint32_t totalLength, uint32_t blockLength,
                                 TPipe* pipeIn) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->tileLength = block_size;
        this->ALIGN_NUM = ALIGN_NUM; 
        this->totalLength = totalLength;
        this->dims = dims;
        this->dimNum = dimNum;
        this->exclusive = exclusive;
        this->reverse = reverse;
        this->blockLength = blockLength;
        this->pipe = pipeIn;
        this->dimSum = dimSum;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ int32_t*)x2, 8); //32byte
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, blockLength);

        pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1)); 
        pipe->InitBuffer(Q_x2, 1, 8 * sizeof(int32_t));
        pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y)); 

        pipe->InitBuffer(B_tmp, this->tileLength * sizeof(TYPE_X1));
        pipe->InitBuffer(B_tmp2, this->tileLength * sizeof(TYPE_X1));
        this->tmp = B_tmp.Get<TYPE_X1>();
        this->tmp2 = B_tmp.Get<TYPE_X1>();
        if constexpr(std::is_same_v<T,int8_t>) {
            pipe->InitBuffer(B_x, this->tileLength * sizeof(half));
            pipe->InitBuffer(B_y, this->tileLength * sizeof(half));
            auto x = B_x.Get<half>();
            Duplicate(x, half(0), this->tileLength);
            Cast(this->tmp, x, RoundMode::CAST_ROUND, this->tileLength);
        } else if constexpr(std::is_same_v<T,float>) {
            Duplicate(this->tmp, float(0), this->tileLength);
        } else if constexpr(std::is_same_v<T,half>) {
            Duplicate(this->tmp, half(0), this->tileLength);
        } else if constexpr(std::is_same_v<T,int32_t>) {
            Duplicate(this->tmp, int32_t(0), this->tileLength);
        }
    }

    __aicore__ inline void PreProcess() {
         CopyInPre();
         ComputePre();
    }

    __aicore__ inline void Process() {
        uint32_t start = 0;
        for (int32_t i = 0; i < totalCumsumCount; i ++) { // 一次cumsum的地址偏移    1    
            start = i * oneCumsumLength;               // 0                       
            bool isFirstCac = true;
            if (reverse) {
                start +=  (currentAixsCount - 1) * oneTimeCacLength; 
                for (int32_t j = currentAixsCount - 1; j >= 0; j --) { // cumsum内的 计算次数
                    for (int k = 0; k < tileNum - 1; k ++) { // 一次计算的切片
                        CopyIn(start, k, this->tileLength,isFirstCac);
                        Compute(k, this->tileLength);
                        CopyOut(start, k, this->tileLength);
                        isFirstCac = false;
                    }
                    CopyIn(start, tileNum - 1, finalLoopLengthAligned,isFirstCac,true,(j == (currentAixsCount - 1)));
                    Compute(tileNum - 1, finalLoopLength,true,finalLoopLengthAligned, (j == (currentAixsCount - 1)));
                    CopyOut(start, tileNum - 1, finalLoopLengthAligned); 
                    start -= oneTimeCacLength;
                    isFirstCac = false; 
                }
            } else {
                for (int32_t j = 0; j < currentAixsCount; j ++) { // cumsum内的 计算次数  2
                    for (int k = 0; k < tileNum - 1; k ++) { // 一次计算的切片  
                        CopyIn(start, k, this->tileLength,isFirstCac);
                        Compute(k, this->tileLength);
                        CopyOut(start, k, this->tileLength);
                        isFirstCac = false;
                    }
                    CopyIn(start, tileNum - 1, finalLoopLengthAligned,isFirstCac); // 0 0 16 1
                    Compute(tileNum - 1, finalLoopLength); // 0 16
                    CopyOut(start, tileNum - 1, finalLoopLengthAligned);  // 0 0 16
                    start += oneTimeCacLength;
                    isFirstCac = false;
                }
            }
        }
    }
 
private: 
    __aicore__ inline void CopyInPre() { 
        LocalTensor<int32_t> x2 = Q_x2.AllocTensor<int32_t>(); 
        DataCopy<int32_t>(x2, Gm_x2[0], 8);
        Q_x2.EnQue(x2); 
    } 
    __aicore__ inline void ComputePre() {
        LocalTensor<int32_t> x2 = Q_x2.DeQue<int32_t>(); 
        this->axis = x2.GetValue(0);  
          if (axis < 0) {
            axis = dimNum + axis;
        }
        this->aixsOffsetLength = dimSum[axis]; // 16
        if (reverse) this->aixsOffsetLength = -this->aixsOffsetLength;
        oneTimeCacLength = 0;
        this->currentAixsCount = dims[axis]; // 2
        oneTimeCacLength = dimSum[axis];  // 16
        if (axis > 0) {
            this->oneCumsumLength = dimSum[axis - 1];  
            this->totalCumsumCount = totalLength / dimSum[axis - 1];
        } else {
            this->totalCumsumCount = 1;
            this->oneCumsumLength = dimSum[0] * dims[0]; 
        }
        this->tileNum = oneTimeCacLength / this->tileLength  + (oneTimeCacLength % this->tileLength ? 1 : 0); // 1
        this->finalLoopLength = oneTimeCacLength - this->tileLength * (this->tileNum - 1); // 16
        this->finalLoopLengthAligned = this->finalLoopLength + (this->finalLoopLength % ALIGN_NUM ? ALIGN_NUM - this->finalLoopLength % ALIGN_NUM : 0);
        this->extraLength = this->finalLoopLengthAligned - this->finalLoopLength;
        Q_x2.FreeTensor(x2);
    }
private:
    __aicore__ inline void CopyIn(uint32_t start, int32_t progress, uint32_t length, bool isFirstCac,bool isLast = false,bool isFirstAxis = false) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>(); 
        if (isFirstCac) {
            if constexpr(std::is_same_v<T,int8_t>) {
                pipe->InitBuffer(B_x, this->tileLength * sizeof(half));
                pipe->InitBuffer(B_y, this->tileLength * sizeof(half));
                auto x = B_x.Get<half>();
                Duplicate(x, half(0), this->tileLength);
                Cast(this->tmp, x, RoundMode::CAST_ROUND, this->tileLength);
            } else if constexpr(std::is_same_v<T,float>) {
                Duplicate(this->tmp, float(0), this->tileLength);
            } else if constexpr(std::is_same_v<T,half>) {
                Duplicate(this->tmp, half(0), this->tileLength);
            } else if constexpr(std::is_same_v<T,int32_t>) {
                Duplicate(this->tmp, int32_t(0), this->tileLength);
            }
        }
        if (exclusive) {
            if (isFirstCac) {
                DataCopy(x1, Gm_y[0], length); 
            } else {
                DataCopy(x1, Gm_x1[start + progress * this->tileLength - aixsOffsetLength], length); 
            }
        } else {
            DataCopy(x1, Gm_x1[start + progress * this->tileLength], length);  
        }
        Q_x1.EnQue(x1); 
    } 
    __aicore__ inline void Compute(int32_t progress, uint32_t length,bool isLast = false, uint32_t lengthAligned = 0, bool isFirstAxis = false) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>(); 
        LocalTensor<TYPE_Y> y_out = Q_y.AllocTensor<TYPE_Y>();
        if constexpr(std::is_same_v<T,int8_t>) {
            LocalTensor<half> y_hf = B_y.Get<half>();
            LocalTensor<half> x_hf = B_x.Get<half>();

            Cast(y_hf, tmp, RoundMode::CAST_NONE, length);
            Cast(x_hf, x1, RoundMode::CAST_NONE, length);
            Add(y_hf, x_hf, y_hf, length);
            Cast(tmp, y_hf, RoundMode::CAST_ROUND, length);
            Cast(y_out, y_hf, RoundMode::CAST_ROUND, length);
  
        } else {
            Add(y_out, x1, tmp, length);
            Add(tmp, x1, tmp, length); 
        }
        Q_y.EnQue(y_out);
        Q_x1.FreeTensor(x1);
    }
    __aicore__ inline void CopyOut(uint32_t start,int32_t progress, uint32_t length) { 
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>(); 
        DataCopy(Gm_y[start + progress * this->tileLength], y, length);
        Q_y.FreeTensor(y); 
    }
 
private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1;
    TQue<QuePosition::VECIN, 1> Q_x2;
    TBuf<QuePosition::VECCALC> B_x, B_y, B_tmp, B_tmp2;  
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;  
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<int32_t> Gm_x2;       
    GlobalTensor<TYPE_Y> Gm_y; 

    LocalTensor<TYPE_X1> tmp,tmp2;

    uint32_t tileLength;
    uint32_t ALIGN_NUM; 
    uint32_t totalLength;
    uint32_t blockLength;
    
    uint32_t* dims;
    uint32_t* dimSum;
    uint32_t dimNum;
    bool exclusive;
    bool reverse;

    uint32_t finalLoopLength;
    uint32_t finalLoopLengthAligned;
    uint32_t tileNum;
    
    int32_t axis = 0;
    uint32_t oneTimeCacLength;
    int32_t aixsOffsetLength;
    uint32_t currentAixsCount;
    uint32_t oneCumsumLength;
    uint32_t totalCumsumCount; 
    uint32_t extraLength;
};


extern "C" __global__ __aicore__ void cumsum(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    Cumsum<DTYPE_X,  DTYPE_Y> op;
    op.Init(x,axis,y,tiling_data.exclusive, tiling_data.reverse, tiling_data.dims,tiling_data.dimSum,
            tiling_data.dimNum,tiling_data.ALIGN_NUM,tiling_data.block_size,
            tiling_data.totalLength,tiling_data.blockLength,&pipe);
    op.PreProcess();
    op.Process();  
}


// a b c d
// 0 a a + b a + b + c
// a a + b a + b + c