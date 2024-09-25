#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X1, typename TYPE_Y> class AsStrided {
    using T = TYPE_X1;
public:
    __aicore__ inline AsStrided() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR size, GM_ADDR stride, GM_ADDR storage_offset, GM_ADDR y,
                                uint32_t vecCount, uint32_t* yShape, uint32_t* yShapeSum, uint32_t* strideArr, uint32_t yDims, uint32_t totalLengthX, uint32_t totalLengthY, uint32_t totalLengthStride,
                                uint32_t ALIGN_NUM, uint32_t block_size) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->tileLength = block_size;
        this->ALIGN_NUM = ALIGN_NUM;

        this->totalLengthX = totalLengthX;
        this->totalLengthY = totalLengthY;
        this->totalLengthStride = totalLengthStride;

        this->blockLengthX = totalLengthX;
        this->blockLengthY = totalLengthY;     
        this->blockLengthStride = totalLengthStride;

        this->extraLengthX = this->blockLengthX % ALIGN_NUM ? ALIGN_NUM - this->blockLengthX % ALIGN_NUM : 0;
        this->extraLengthY = this->blockLengthY % ALIGN_NUM ? ALIGN_NUM - this->blockLengthY % ALIGN_NUM : 0;
        this->extraLengthStride = this->blockLengthStride % ALIGN_NUM ? ALIGN_NUM - this->blockLengthStride % ALIGN_NUM : 0;

        this->blockLengthX = this->blockLengthX + this->extraLengthX; // 11472
        this->blockLengthY = this->blockLengthY + this->extraLengthY; // 11472  
        this->blockLengthStride = this->blockLengthStride + this->extraLengthStride;


        auto bufferlengthX = this->blockLengthX;  
        auto bufferlengthY = this->blockLengthY; 
        auto bufferlengthStride = this->blockLengthStride; 

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, bufferlengthX);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, bufferlengthY); // 先不改等Y实现后在该
        Gm_stride.SetGlobalBuffer((__gm__ int32_t*)stride, bufferlengthStride);
        Gm_storage_offset.SetGlobalBuffer((__gm__ int32_t*)storage_offset, 8); //就一个值向上直接补齐到32字节即可

        this->tileNumX = this->blockLengthX / this->tileLength + (this->blockLengthX % this->tileLength > 0);
        this->tileNumY = this->blockLengthY / this->tileLength + (this->blockLengthY % this->tileLength > 0);
        this->tileNumStride = this->blockLengthStride / this->tileLength + (this->blockLengthStride % this->tileLength > 0);
        
        this->strideArr = strideArr;
        this->yShape = yShape;
        this->yShapeSum = yShapeSum;
        this->vecCount = vecCount;
        this->yDims = yDims;

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(Q_stride, BUFFER_NUM, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(Q_storage_offset, 1, 8 * sizeof(int32_t));  //32字节对齐保证
 

    }
    //获取oriPos 和 stride数组
    __aicore__ inline void PreProcess() {
        CopyInOffset();
        ComputeOffset();
        int32_t loopCountStride = this->tileNumStride;
         for (int32_t i = 0; i < loopCountStride - 1; i++) {
            CopyInStride(i, this->tileLength);
            ComputeStride(i, this->tileLength, 0); // indicies存储于transferBuffer
        }
        auto lengthStride = this->blockLengthStride - this->tileLength * (loopCountStride - 1);
        auto cacLengthStride = this->totalLengthStride - this->tileLength * (loopCountStride - 1);
        CopyInStride(loopCountStride - 1, lengthStride);
        ComputeStride(loopCountStride - 1, cacLengthStride, this->tileLength * (loopCountStride - 1));
    
    }

    __aicore__ inline void Process() {
        /*
            yShape = [3,4,5,6]
            yShapeSum = [20,5,1,0]
            strideArr = [1,2,1,2]
        */
        uint32_t lstIdx = this->yDims - 1;
        uint32_t eleCount = yShape[lstIdx];  //32
        for (int i = 0; i < this->vecCount; i ++) { // 16 
            uint32_t startY = i * eleCount; // 0
            uint32_t currentPos = oriPos; // 58
            for (int j = 0; j < yDims - 1; j ++) { // 0
                currentPos += ((i / yShapeSum[j]) % yShape[j]) * strideArr[j]; //0
            }
            if (strideArr[lstIdx] == uint32_t(1)){   
                //ASSERT("MAKE IT");
                uint32_t currentTotalLength = yShape[lstIdx];
                uint32_t currentExtraLength = currentTotalLength % ALIGN_NUM ? ALIGN_NUM - currentTotalLength % ALIGN_NUM : 0;
                uint32_t currentAlignedTotalLength = currentTotalLength + currentExtraLength;
                uint32_t loopCount = currentAlignedTotalLength / this->tileLength + (currentAlignedTotalLength % this->tileLength > 0);
                for (int32_t j = 0; j < loopCount - 1; j ++) {
                    CopyIn_1(j, currentPos, this->tileLength);
                    Compute_1(j,this->tileLength);
                    CopyOut_1(j, startY, this->tileLength);
                }
                uint32_t currentLength = currentAlignedTotalLength - (loopCount - 1) * this->tileLength;
                uint32_t currentLengthAligned = currentLength + (currentLength % ALIGN_NUM ? ALIGN_NUM - currentLength % ALIGN_NUM : 0);
                CopyIn_1(loopCount - 1, currentPos, currentLengthAligned);
                Compute_1(loopCount - 1, currentLengthAligned);
                CopyOut_1(loopCount - 1, startY, currentLengthAligned);
                
            } else {
                // 0 1 2 3 4 5 6 7
                // 0 3 6
                // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
                uint32_t tileEleCount = (this->tileLength - 1) / strideArr[lstIdx] + 1; // 一个tile能处理多少个元素
                uint32_t strideLeft = (this->tileLength - 1) % strideArr[lstIdx];
                uint32_t posIncrement = strideArr[lstIdx] - (strideLeft + 1); 
                if (tileEleCount <= eleCount) {        
                    for (int j = 0; j <= eleCount - tileEleCount; j += tileEleCount) {
                        CopyIn(currentPos, this->tileLength);
                        Compute(strideArr[lstIdx], tileEleCount);
                        CopyOut(startY + j,  tileEleCount + (tileEleCount % ALIGN_NUM ? ALIGN_NUM - (tileEleCount % ALIGN_NUM) : 0));
                        currentPos += this->tileLength + posIncrement;
                    }
                }
                uint32_t left = eleCount % tileEleCount;
                if (left) {
                    uint32_t totalLength = left  * strideArr[lstIdx];
                    uint32_t totalLengthAliged = totalLength + (totalLength % ALIGN_NUM ? ALIGN_NUM - (totalLength % ALIGN_NUM) : 0);
                    CopyIn(currentPos, totalLengthAliged);
                    Compute(strideArr[lstIdx], left);
                    CopyOut(startY + eleCount - left, left + (left % ALIGN_NUM ? ALIGN_NUM - (left % ALIGN_NUM) : 0));
                }
            }
        }
    }
private:
    __aicore__ inline void CopyInOffset() {
        LocalTensor<int32_t> offSetLocal = Q_storage_offset.AllocTensor<int32_t>();
        DataCopy(offSetLocal, Gm_storage_offset[0], 8);
        Q_storage_offset.EnQue(offSetLocal);
    }
    __aicore__ inline void ComputeOffset() {
        LocalTensor<int32_t> offSetLocal = Q_storage_offset.DeQue<int32_t>();
        this->oriPos = offSetLocal.GetValue(0);
        Q_storage_offset.FreeTensor(offSetLocal);
    }
private:
    __aicore__ inline void CopyInStride(int32_t progress, uint32_t length) {
        LocalTensor<int32_t> strideLocal = Q_stride.AllocTensor<int32_t>();
        DataCopy(strideLocal, Gm_stride[progress * this->tileLength], length);
        Q_stride.EnQue(strideLocal);
    }
    __aicore__ inline void ComputeStride(int32_t progress, uint32_t length, uint32_t startPos) {
        LocalTensor<int32_t> strideLocal = Q_stride.DeQue<int32_t>();
        for (int i = 0; i < length; i ++) {
            this->strideArr[startPos + i] = strideLocal.GetValue(i);
        }
        Q_stride.FreeTensor(strideLocal);
    }
private:
    __aicore__ inline void CopyIn_1(int32_t progress, uint32_t start, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        DataCopy(x1, Gm_x1[start + progress * this->tileLength], length);
        Q_x1.EnQue(x1);
    }
    __aicore__ inline void Compute_1(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        Adds(y, x1, TYPE_X1(0), length); 
        Q_y.EnQue(y);
        Q_x1.FreeTensor(x1);
    }
    
    __aicore__ inline void CopyOut_1(int32_t progress, uint32_t start, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[start + progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    __aicore__ inline void CopyIn(uint32_t start, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        DataCopy(x1, Gm_x1[start], length);
        Q_x1.EnQue(x1);
    }
    __aicore__ inline void Compute(uint32_t strideLength, uint32_t count) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        for (int i = 0; i < count; i ++) {
            y.SetValue(i, x1.GetValue(i * strideLength));
        }
        Q_x1.FreeTensor(x1);
        Q_y.EnQue(y);
    }
    __aicore__ inline void CopyOut(uint32_t start, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[start], y, length);
        Q_y.FreeTensor(y);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_stride, Q_storage_offset;  
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;  
    GlobalTensor<TYPE_X1> Gm_x1; 
    GlobalTensor<int32_t> Gm_stride;
    GlobalTensor<int32_t> Gm_storage_offset;      
    GlobalTensor<TYPE_Y> Gm_y;


    uint32_t tileLength;
    uint32_t ALIGN_NUM;
    uint32_t tileNumX;
    uint32_t tileNumY;
    uint32_t tileNumStride;

    uint32_t totalLengthX;
    uint32_t totalLengthY;
    uint32_t totalLengthStride;

    uint32_t blockLengthX; 
    uint32_t blockLengthY;
    uint32_t blockLengthStride;

    uint32_t extraLengthX;
    uint32_t extraLengthY;
    uint32_t extraLengthStride;
    
    int32_t oriPos = 0;
    uint32_t yDims = 0;
    uint32_t vecCount = 0;
    uint32_t* yShape;
    uint32_t* strideArr;
    uint32_t* yShapeSum;
};

extern "C" __global__ __aicore__ void as_strided(GM_ADDR x, GM_ADDR size, GM_ADDR stride, GM_ADDR storage_offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    /*
    x1, size, stride, storage_offset, y, vecCount, yShape, yShapeSum, strideArr, yDims
    totalLengthX, totalLengthY,
    totalLengthStride,
    uint32_t ALIGN_NUM, uint32_t block_size
    */
    AsStrided<DTYPE_X, DTYPE_Y> op;
    op.Init(x,size,stride,storage_offset,y,tiling_data.vecCount,tiling_data.yShape,tiling_data.yShapeSum,tiling_data.strideArr,tiling_data.yDims,
            tiling_data.totalLengthX,tiling_data.totalLengthY,tiling_data.totalLengthStride,tiling_data.ALIGN_NUM,tiling_data.block_size);
    op.PreProcess();
    op.Process();            
}