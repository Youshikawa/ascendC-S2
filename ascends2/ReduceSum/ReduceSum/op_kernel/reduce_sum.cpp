#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MAX_DIM_COUNT = 24;

/*检查是否gm访问越界*/
template<typename TYPE_X1, typename TYPE_Y> class KernelReduceSum {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelReduceSum() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,bool ignore_nan,uint32_t blockLengthY, uint32_t* dims, uint32_t* dimSum, uint32_t dimNum,
                                 uint32_t ALIGN_NUM, uint32_t block_size, uint32_t totalLength, uint32_t blockLength,uint32_t totalLengthAxes, uint32_t blockLengthAxes,
                                 TPipe* pipeIn) 
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        {
            this->pipe = pipeIn;
            this->tileLength = block_size;
            this->ALIGN_NUM = ALIGN_NUM; 
            this->totalLength = totalLength;
            this->blockLength = blockLength;
            this->totalLengthAxes = totalLengthAxes;
            this->blockLengthAxes = blockLengthAxes;

            this->ignore_nan = ignore_nan; 
            this->dims = dims;
            this->dimNum = dimNum;
            this->dimSum = dimSum;        
        } // 初始化赋值

        {
            Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, blockLength);
            Gm_x2.SetGlobalBuffer((__gm__ int32_t*)x2, blockLengthAxes); //32byte 4字节 8个最少
            Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, blockLength);

            pipe->InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1)); 
            pipe->InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(int32_t));
            pipe->InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y)); 
            pipe->InitBuffer(Q_y2, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));   
            pipe->InitBuffer(Q_x3, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

            pipe->InitBuffer(B_x, this->tileLength * sizeof(float)); 
            pipe->InitBuffer(B_x2, this->tileLength * sizeof(half)); 
            pipe->InitBuffer(B_y, this->tileLength * sizeof(float)); 
            pipe->InitBuffer(B_tmp, this->tileLength * sizeof(float)); 
            pipe->InitBuffer(B_workBuffer1, this->tileLength * sizeof(float)); 
            this->tmp = B_tmp.Get<float>(); 

            if (std::is_same_v<T,int8_t>)
            {    
                pipe->InitBuffer(B_x, this->tileLength * sizeof(half));  
            }
        } // GM及UB内存初始化
    }

    __aicore__ inline void PreProcess() // 将axes信息提取并存储
    {
        int32_t loopCount = this->totalLengthAxes / this->tileLength + (this->totalLengthAxes % this->tileLength ? 1 : 0); 
        for (int32_t i = 0; i < loopCount - 1; i ++)
        {
            CopyInPre(i, this->tileLength);
            ComputePre(i, this->tileLength);
        }
        uint32_t finalLoopLength = this->totalLengthAxes - (loopCount - 1) * this->tileLength;
        uint32_t finalLoopLengthAligned = this->blockLengthAxes - (loopCount - 1) * this->tileLength;
        CopyInPre(loopCount - 1, finalLoopLengthAligned);
        ComputePre(loopCount - 1, finalLoopLength);
 
    }
    __aicore__ inline void Process() // 进一步划分axes做出真正计算
    {  
        int32_t cacedLength = 1;
        int32_t i = dimNum - 1; 
        for (; i >= 0; i --) {
            if (cacAxes[i] == 0) break;

        }
        i += 1;
        bool x2y = true;
        if (i <= (dimNum - 1)) {
            int32_t oneTimeCacLength = i == 0 ? dimSum[0] * dims[0] : dimSum[i - 1];
            continuedProcess(oneTimeCacLength, oneTimeCacLength, totalLength / oneTimeCacLength);
            cacedLength = oneTimeCacLength;
            x2y = false;
            dimSeter(i, dimNum - 1);
        }  
        for (int32_t j = i - 1; j >= 0; j --) {
            if (cacAxes[j]) {
                int32_t k = j;
                for (; k >= 0; k --) if (cacAxes[k] == 0) break;
                k += 1;
                int32_t currentLength = dimSum[k]; 
                if (k == 0) currentLength = dimSum[k] * dims[0];
                else currentLength = dimSum[k - 1];
                offsetProcess(dimSum[j], currentLength / dimSum[j],currentLength,totalLength / currentLength,x2y);
                x2y = !x2y;
                dimSeter(k, j);
                j = k;
            }
        }
        // 2   2 6
        // 12 6 1
        if (x2y) {

        }
    }
private: // 预处理阶段
    __aicore__ inline void CopyInPre(int32_t progress, uint32_t length) { 
        LocalTensor<int32_t> x2 = Q_x2.AllocTensor<int32_t>(); 
        DataCopy<int32_t>(x2, Gm_x2[progress * this->tileLength], length);
        Q_x2.EnQue(x2); 
    } 
    __aicore__ inline void ComputePre(int32_t progress, uint32_t length) {
        LocalTensor<int32_t> x2 = Q_x2.DeQue<int32_t>();  
        for (int32_t i = 0; i < length; i ++)
        {
            int32_t t = x2.GetValue(i);
            if (t < 0) t += dimNum;
            cacAxes[t] = true;
        } 
        Q_x2.FreeTensor(x2); 
    }
 
private: // 连续处理阶段
    __aicore__ inline void continuedProcess(int32_t oneTimeCacLength, int32_t oneReduceSumLength, int32_t reduceSumCount) {
        int32_t loopCount = oneTimeCacLength / this->tileLength + (oneTimeCacLength % this->tileLength ? 1 : 0);
        uint32_t finalLoopLength = oneTimeCacLength - (loopCount - 1) * this->tileLength;
        uint32_t finalLoopLengthAligned = finalLoopLength + (finalLoopLength % ALIGN_NUM ? ALIGN_NUM - (finalLoopLength % ALIGN_NUM) : 0);
        int32_t start = 0; 
        for (int32_t i = 0; i < reduceSumCount; i ++) {
            start = i * oneReduceSumLength;
            
            for (int32_t j = 0; j < loopCount - 1; j ++) {
                continuedCopyIn(start, j, this->tileLength, j == 0);
                continuedCompute(start, j, this->tileLength, false);
                continuedCopyOut(i, ALIGN_NUM, false);
            }
            continuedCopyIn(start, loopCount - 1, finalLoopLengthAligned, (loopCount - 1) == 0);   
            continuedCompute(start, loopCount - 1, finalLoopLength, true);
            continuedCopyOut(i, ALIGN_NUM, true);
        }
    }
    __aicore__ inline void continuedCopyIn(int32_t start, int32_t progress, uint32_t length, bool isFirst) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        DataCopy(x1, Gm_x1[start + progress * this->tileLength], length);
        if (isFirst) {
            continuedTransferValue = 0;
        }
        Q_x1.EnQue(x1);
    }
    __aicore__ inline void continuedCompute(int32_t start, int32_t progress, uint32_t length, bool isLast) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>(); 
        auto workBuffer1 = B_workBuffer1.Get<float>();
        if constexpr (std::is_same_v<T, float>) {
            ReduceSum(tmp, x1, workBuffer1, length);
            Adds(tmp, tmp, continuedTransferValue, 1);
            continuedTransferValue = tmp.GetValue(0);  
            if (isLast) y.SetValue(0,continuedTransferValue);
        } else if constexpr(std::is_same_v<T, int8_t>) {
            auto x1_fl = B_x.Get<float>();
            auto x1_hf = B_x2.Get<half>(); 
            Cast(x1_hf, x1, RoundMode::CAST_NONE, length);
            Cast(x1_fl, x1_hf, RoundMode::CAST_NONE, length);
            ReduceSum(tmp, x1_fl, workBuffer1, length);
            Adds(tmp, tmp, float(continuedTransferValue), 1);
            continuedTransferValue = tmp.GetValue(0);  
            if (isLast) {
                y.SetValue(0,int8_t(half(continuedTransferValue)));    
            }
        } else {
            auto x1_fl = B_x.Get<float>();
            Cast(x1_fl, x1, RoundMode::CAST_NONE, length);
            ReduceSum(tmp, x1_fl, workBuffer1, length);
            Adds(tmp, tmp, continuedTransferValue, 1);
            continuedTransferValue = tmp.GetValue(0);  
            if (isLast) {
                Cast(x1, tmp, RoundMode::CAST_ROUND, length);
                y.SetValue(0,tmp.GetValue(0));    
            }
        }
        Q_y.EnQue(y);
        Q_x1.FreeTensor(x1);
    }
    __aicore__ inline void continuedCopyOut(uint32_t start, uint32_t length, bool isLast) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        if(isLast) DataCopy(Gm_y[start], y, length);
        Q_y.FreeTensor(y);
    }
private: // 偏移处理阶段
    __aicore__ inline void offsetProcess(int32_t oneTimeCacLength, int32_t innerCacCount, int32_t oneReduceSumLength, 
                                        int32_t reduceSumCount, bool x2y) { // 6 2 12 2 1
        int32_t loopCount = oneTimeCacLength / this->tileLength + (oneTimeCacLength % this->tileLength ? 1 : 0);
        uint32_t finalLoopLength = oneTimeCacLength - (loopCount - 1) * this->tileLength;
        uint32_t finalLoopLengthAligned = finalLoopLength + (finalLoopLength % ALIGN_NUM ? ALIGN_NUM - (finalLoopLength % ALIGN_NUM) : 0);
        uint32_t start = 0;
        for (int32_t k = 0; k < loopCount - 1; k ++) {
            for (int32_t i = 0; i < reduceSumCount; i ++) {
                start = i * oneReduceSumLength;
                for (int32_t j = 0; j < innerCacCount; j ++) { // j 为最后一个时copyout
                    offsetCopyIn(start, k, this->tileLength, x2y, j == 0);
                    offsetCompute(k, this->tileLength, x2y, (j + 1) == (innerCacCount));
                    offsetCopyOut(i * oneTimeCacLength, k, this->tileLength, x2y, (j + 1) == (innerCacCount));
                    start += oneTimeCacLength;
                }
            }
        }
        for (int32_t i = 0; i < reduceSumCount; i ++) {
            start = i * oneReduceSumLength;
            for (int32_t j = 0; j < innerCacCount; j ++) {
                offsetCopyIn(start, loopCount - 1, finalLoopLengthAligned, x2y, j == 0);
                offsetCompute(loopCount - 1, finalLoopLength, x2y, (j + 1) == (innerCacCount));
                offsetCopyOut(i * oneTimeCacLength, loopCount - 1, finalLoopLengthAligned, x2y, (j + 1) == (innerCacCount));
                start += oneTimeCacLength;
            }
        }

    }
    __aicore__ inline void offsetCopyIn(uint32_t start, int32_t progress, uint32_t length, bool x2y, bool isFirst) {
        if (isFirst) Duplicate(tmp, float(0), length);
        if (x2y) {
            LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
            DataCopy(x1, Gm_x1[start + progress * this->tileLength], length);
            Q_x1.EnQue(x1);
        } else {
            LocalTensor<TYPE_Y> y2 = Q_y2.AllocTensor<TYPE_Y>();
            DataCopy(y2, Gm_y[start + progress * this->tileLength], length);
            Q_y2.EnQue(y2);
        }
    }
    __aicore__ inline void offsetCompute(int32_t progress, uint32_t length, bool x2y, bool isLast) {
        if (x2y) {
            LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
            LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
            if constexpr (std::is_same_v<T,float>) {
                Add(tmp, tmp, x1, length);
                if (isLast) {
                    Adds(y, tmp, TYPE_Y(0), length);
                }
            } else if constexpr(std::is_same_v<T, int8_t>) {
                auto x1_fl = B_x.Get<float>();
                auto x1_hf = B_x2.Get<half>();
                Cast(x1_hf, x1, RoundMode::CAST_NONE, length);
                Cast(x1_fl, x1_hf, RoundMode::CAST_NONE, length); 
                Add(tmp, tmp, x1_fl, length);
                if (isLast) {
                    Cast(x1_hf, tmp, RoundMode::CAST_ROUND, length);
                    Cast(y, x1_hf, RoundMode::CAST_ROUND, length);
                }
            } else { 
                auto x1_fl = B_x.Get<float>();
                Cast(x1_fl, x1, RoundMode::CAST_NONE, length);
                Add(tmp, tmp, x1_fl, length);
                if (isLast) {
                    Cast(y, tmp, RoundMode::CAST_ROUND, length);
                }
            }
            Q_y.EnQue(y);
            Q_x1.FreeTensor(x1);
        } else {
            LocalTensor<TYPE_Y> y2 = Q_y2.DeQue<TYPE_Y>();
            LocalTensor<TYPE_X1> x3 = Q_x3.AllocTensor<TYPE_X1>();
            if constexpr (std::is_same_v<T,float>) {
                Add(tmp, tmp, y2, length);
                if (isLast) {
                    Adds(x3, tmp, TYPE_X1(0), length);
                }
            } else if constexpr(std::is_same_v<T, int8_t>) {
                auto x1_fl = B_x.Get<float>();
                auto x1_hf = B_x2.Get<half>();
                Cast(x1_hf, y2, RoundMode::CAST_NONE, length);
                Cast(x1_fl, x1_hf, RoundMode::CAST_NONE, length); 
                Add(tmp, tmp, x1_fl, length);
                if (isLast) {
                    Cast(x1_hf, tmp, RoundMode::CAST_ROUND, length);
                    Cast(x3, x1_hf, RoundMode::CAST_ROUND, length);
                }
            } else {
                auto x1_fl = B_x.Get<float>();
                Cast(x1_fl, y2, RoundMode::CAST_NONE, length);
                Add(tmp, tmp, x1_fl, length);
                if (isLast) {
                    Cast(x3, tmp, RoundMode::CAST_ROUND, length);
                }
            }
            Q_x3.EnQue(x3);
            Q_y2.FreeTensor(y2);
        }
    }
    __aicore__ inline void offsetCopyOut(uint32_t start, int32_t progress, uint32_t length, bool x2y, bool isLast) {
        if (x2y) {
            LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
            if (isLast) {
                DataCopy(Gm_y[start + progress * this->tileLength], y, length);
            }
            Q_y.FreeTensor(y);
        } else {
            LocalTensor<TYPE_X1> x3 = Q_x3.DeQue<TYPE_X1>();
            if (isLast) {
                DataCopy(Gm_x1[start + progress * this->tileLength], x3, length);
            }
            Q_x3.FreeTensor(x3);
        }
    }
private:
    __aicore__ inline void SetZero(LocalTensor<T>& x, uint32_t length) 
    {
        if constexpr (std::is_same_v<T,int8_t>) {
            auto zero_tmp = B_x.Get<half>();
            Duplicate(zero_tmp, half(0), length);
            Cast(x, zero_tmp, RoundMode::CAST_ROUND, length);
        } else {
            Duplicate(x, T(0), length);
        }
    } 
    
    __aicore__ inline void dimSeter(int32_t start,int32_t end) 
    {
        int32_t idx = 0;
        for (int32_t i = 0; i < dimNum; i ++) {
            if (i == start) {
                dims[idx ++] = 1;
                i = end;
            } else {
                dims[idx ++] = dims[i];
            }
        }
        dimNum = idx;
        dimSum[dimNum] = 1;
        for (int i = dimNum - 1; i >= 0; i --) {
            dimSum[i] = dimSum[i + 1];
            dimSum[i] *= dims[i + 1];
        }
    } 

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2, Q_y2; 
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y, Q_x3;  
    TBuf<QuePosition::VECCALC> B_x, B_y, B_tmp, B_workBuffer1, B_x2;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<int32_t> Gm_x2;       
    GlobalTensor<TYPE_Y> Gm_y; 
    
    LocalTensor<float> tmp;

    float continuedTransferValue = 0;

    uint32_t tileLength;
    uint32_t ALIGN_NUM; 
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t totalLengthAxes;
    uint32_t blockLengthAxes;

    bool cacAxes[MAX_DIM_COUNT] {0};
    uint32_t aixsOffsetLength[MAX_DIM_COUNT] {0};
    uint32_t oneTimeCacLength[MAX_DIM_COUNT] {0};
    uint32_t oneReduceSumLength[MAX_DIM_COUNT] {0};
    uint32_t cacCount = 0;

    uint32_t* dims;
    uint32_t* dimSum;
    uint32_t dimNum;
    bool ignore_nan;
  
 

};

extern "C" __global__ __aicore__ void reduce_sum(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelReduceSum<DTYPE_X,DTYPE_Y> op;
    op.Init(x,axes,y,tiling_data.ignore_nan,tiling_data.blockLengthY, tiling_data.dims,tiling_data.dimSum,
            tiling_data.dimNum,tiling_data.ALIGN_NUM,tiling_data.block_size,
            tiling_data.totalLength,tiling_data.blockLength,tiling_data.totalLengthAxes,tiling_data.blockLengthAxes,&pipe);
    op.PreProcess();
     op.Process();
} 