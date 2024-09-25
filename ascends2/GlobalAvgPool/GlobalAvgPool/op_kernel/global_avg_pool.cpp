#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

/*
    思路：
    1. yshape 和 xshape不同，因为最好要拷贝到y，且每次x计算仅计算出一个y的元素， 不妨循环NC，设置中间result，等result满或计算结束在进行y的数据搬运
    2. 优化思路 当为NC*1的情况直接搬运设置tilingkey好了

*/

template<typename TYPE_X1, typename TYPE_Y> class GlobalAvgPool {
    using T = TYPE_X1;

public:
    __aicore__ inline GlobalAvgPool() {}
    __aicore__ inline void Init(GM_ADDR x1,GM_ADDR y,float chanelLengthFloat, uint32_t totalLengthX, uint32_t blockLengthX,uint32_t ALIGN_NUM, uint32_t block_size,
                                uint32_t totalLengthY, uint32_t blockLengthY,
                                uint32_t chanelCount,uint32_t chanelLength,uint32_t chanelLengthAligned,
                                uint32_t tileNumChanel,uint32_t finalLoopLengthChanel,uint32_t finalLoopCacLengthChanel) {
        this->tileLength = block_size;
        this->ALIGN_NUM = ALIGN_NUM;

        this->totalLengthX = totalLengthX;
        this->blockLengthX = blockLengthX;

        this->totalLengthY = totalLengthY;
        this->blockLengthY = blockLengthY;

        this->chanelCount = chanelCount;
        this->chanelLength = chanelLength;
        this->chanelLengthAligned = chanelLengthAligned;
        this->tileNumChanel = tileNumChanel;
        this->finalLoopLengthChanel = finalLoopLengthChanel; // fll
        this->finalLoopCacLengthChanel = finalLoopCacLengthChanel; //flc

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, blockLengthX);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, blockLengthY);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_transferSumBuffer, this->tileLength * sizeof(float)); 
        pipe.InitBuffer(B_workBuffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_tmpBuffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_divBuffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_int16, this->tileLength * sizeof(int16_t));

        this->chanelDiv = B_divBuffer.Get<float>();
        this->transferSum = B_transferSumBuffer.Get<float>(); 
        Duplicate(this->transferSum, float(0), this->tileLength);
        Duplicate(this->chanelDiv, chanelLengthFloat, this->tileLength);

        if constexpr(std::is_same_v<T, float>){
        } else { 
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        uint32_t startX = 0; // 搬运X的位置 
        for (int32_t i = 0; i < this->chanelCount; i ++) { // 1
            int32_t loopCount = this->tileNumChanel; // 1
            for (int32_t j = 0; j < loopCount - 1; j ++) { 
                CopyIn(j, startX, this->tileLength);
                Compute(j, this->tileLength);
                CopyOut(j, this->tileLength);
            }
            CopyIn(loopCount - 1, startX, this->finalLoopLengthChanel); // 0 0 8
            Compute(loopCount - 1, this->finalLoopCacLengthChanel, true); // 0 8 1 0 1
            if constexpr(std::is_same_v<T, float>) CopyOut(i, 8, true);
            else CopyOut(i, 16, true);
            startX += this->chanelLength;
        }
        
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t start, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        DataCopy(x1, Gm_x1[start + progress * this->tileLength], length);
        Q_x1.EnQue(x1);
    }   
    __aicore__ inline void Compute(int32_t progress, uint32_t length, bool isLast = false) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        auto workBuffer = B_workBuffer.Get<float>();
        auto tmpBuffer = B_tmpBuffer.Get<float>();
        LocalTensor<float> x;
        if constexpr(std::is_same_v<T, float>) {
            x = x1; 
        } else { // half
            auto fl_x = B_x1.Get<float>();
            Cast(fl_x, x1, RoundMode::CAST_NONE, length);
            x = fl_x; 
        }
        ReduceSum(tmpBuffer, x, workBuffer, length); // sum
        if constexpr(std::is_same_v<T, half>) {
            Cast(y, tmpBuffer, RoundMode::CAST_ROUND, length);
            Cast(tmpBuffer, y, RoundMode::CAST_NONE, length);
        }
        Add(this->transferSum, this->transferSum, tmpBuffer, 1); //sum
        if constexpr(std::is_same_v<T, half>) {
            Cast(y, this->transferSum, RoundMode::CAST_ROUND, 1);
            Cast(this->transferSum, y, RoundMode::CAST_NONE, 1);
        }
        if (isLast) {
            if constexpr(std::is_same_v<T,float>) {
                Mul(this->transferSum, this->transferSum, this->chanelDiv, 1); // avg
                Adds(y, this->transferSum, float(0), 1);
            } else {
                Reciprocal(tmpBuffer, this->chanelDiv, 1);
                Div(this->transferSum, this->transferSum, tmpBuffer, 1); // avg
            
                Cast(y, this->transferSum, RoundMode::CAST_ROUND, length); 
                // Compare(bits, this->transferSum, tmpBuffer, CMPMODE::GE, 64);
                // if (bits.GetValue(0)) {
                //     Adds(y,y,half(-0.0009),1);
                // } else {
                // Adds(y,y,half(0.001),length);
                // }
            }
            this->transferSum.SetValue(0, 0);
        }
        Q_x1.FreeTensor(x1);
        Q_y.EnQue(y);
    }   
    __aicore__ inline void CopyOut(uint32_t start, uint32_t length, bool isLast = false) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        if (isLast) DataCopy(Gm_y[start], y, length);
        Q_y.FreeTensor(y);
    }   

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1;
    TBuf<QuePosition::VECCALC> B_x1, B_transferSumBuffer, B_workBuffer, B_tmpBuffer, B_divBuffer, B_bits, B_int16; // 全部为float  
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;  
    GlobalTensor<TYPE_X1> Gm_x1;       
    GlobalTensor<TYPE_Y> Gm_y;  
    LocalTensor<float> transferSum, chanelDiv;

    uint32_t tileLength;
    uint32_t ALIGN_NUM;
 
    uint32_t totalLengthX;
    uint32_t blockLengthX;

    uint32_t totalLengthY;
    uint32_t blockLengthY;
      

    uint32_t chanelCount;
    uint32_t chanelLength;
    uint32_t chanelLengthAligned;
    uint32_t tileNumChanel;
    uint32_t finalLoopLengthChanel;
    uint32_t finalLoopCacLengthChanel; 

};

extern "C" __global__ __aicore__ void global_avg_pool(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GlobalAvgPool<DTYPE_X, DTYPE_Y> op;
    /*
    GM_ADDR x1,GM_ADDR y,uint32_t totalLengthX, uint32_t blockLengthX,uint32_t ALIGN_NUM, uint32_t block_size,
                                uint32_t totalLengthY, uint32_t blockLengthY,
                                uint32_t chanelCount,uint32_t chanelLength,uint32_t chanelLengthAligned,
                                uint32_t tileNumChanel,uint32_t finalLoopLengthChanel,uint32_t finalLoopCacLengthChanel
    
     */
    op.Init(x, y, tiling_data.chanelLengthFloat, tiling_data.totalLengthX, tiling_data.blockLengthX, tiling_data.ALIGN_NUM, tiling_data.block_size,
            tiling_data.totalLengthY, tiling_data.blockLengthY,
            tiling_data.chanelCount, tiling_data.chanelLength, tiling_data.chanelLengthAligned,
            tiling_data.tileNumChanel, tiling_data.finalLoopLengthChanel, tiling_data.finalLoopCacLengthChanel);
   op.Process(); 
}