#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

// 任务：精度不够 拟合一下

// min,max != 0, 0 
// 实现1 直接getvalue setvalue
// 实现2 将x getvalue的值放入mask 再通过mask操作y
template<typename TYPE_X1, typename TYPE_Y> class Histogram {
    using T = TYPE_X1;
public:
    __aicore__ inline Histogram() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, bool cacMinMax, int32_t bins, float min, float max, float area, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t totalLengthX, uint32_t totalLengthY) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->tileLength = block_size;
        this->ALIGN_NUM = ALIGN_NUM; 

        this->totalLengthX = totalLengthX;
        this->totalLengthY = totalLengthY;
        this->blockLengthX = totalLengthX;
        this->blockLengthY = totalLengthY;
        this->oriLengthX = this->blockLengthX;
        this->oriLengthY = this->blockLengthY;      
        this->extraLengthX = this->blockLengthX % ALIGN_NUM ? ALIGN_NUM - this->blockLengthX % ALIGN_NUM : 0;
        this->extraLengthY = this->blockLengthY % ALIGN_NUM ? ALIGN_NUM - this->blockLengthY % ALIGN_NUM : 0;
        this->blockLengthX = this->blockLengthX + this->extraLengthX; // 11472
        this->blockLengthY = this->blockLengthY + this->extraLengthY; // 11472  
 
        this->bins = bins;  
        this->min = min;
        this->max = max;
        this->area = area;
 
        auto bufferlengthX = this->blockLengthX;  
        auto bufferlengthY = this->blockLengthY;  

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, bufferlengthX);  
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, bufferlengthY); // 先不改等Y实现后在该

        this->tileNumX = this->blockLengthX / this->tileLength + (this->blockLengthX % this->tileLength > 0);
        this->tileNumY = this->blockLengthY / this->tileLength + (this->blockLengthY % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));  
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));  

        if constexpr(std::is_same_v<T, half>) {
            pipe.InitBuffer(B_flx, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_fly, this->tileLength * sizeof(float));
        } else if constexpr(std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_flx, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_fly, this->tileLength * sizeof(float));
        } else { // float
        }  
        pipe.InitBuffer(B_transInt32Indices, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(B_zeros, this->tileLength * sizeof(float)); 
        pipe.InitBuffer(B_int32y, this->tileLength * sizeof(int32_t));
        pipe.InitBuffer(B_tmp2, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_tmp, this->tileLength * sizeof(float)); 
        auto zeros = B_zeros.Get<float>();
        Duplicate(zeros, float(0), this->tileLength);
        this->zeros = zeros;

        if (cacMinMax) {
            pipe.InitBuffer(B_maxBuffer, this->tileLength);
            pipe.InitBuffer(B_minBuffer, this->tileLength);
        }

        this->cacMinMax = cacMinMax;
        /*
            初始化bin_edge数组
        */
    }

    __aicore__ inline void PreProcess() {
        int32_t loopCountX = this->tileNumX;
         for (int32_t i = 0; i < loopCountX - 1; i++) {
            CopyInX(i, this->tileLength);
            ComputeX_1(i, this->tileLength, this->tileLength); // indicies存储于transferBuffer
        }
        auto lengthX = this->blockLengthX - this->tileLength * (loopCountX - 1);
        auto cacLengthX = this->oriLengthX - this->tileLength * (loopCountX - 1);
        CopyInX(loopCountX - 1, lengthX);
        ComputeX_1(loopCountX - 1, cacLengthX, (lengthX + (this->ALIGN_NUM * 8) - 1) / (this->ALIGN_NUM * 8) * (this->ALIGN_NUM * 8));
        // if constexpr(std::is_same_v<T, float>)
        //     DataCopy(Gm_y[0], minBuffer, 1);
    }

    __aicore__ inline void Process() {
        int32_t loopCountX = this->tileNumX;
        int32_t loopCountY = this->tileNumY;
        bool flag = true;
        for (int32_t i = 0; i < loopCountX - 1; i++) {
            CopyInX(i, this->tileLength);
            ComputeX(i, this->tileLength, this->tileLength); // indicies存储于transferBuffer
            for (int32_t j = 0; j < loopCountY - 1; j ++) {
                CopyInY(j, this->tileLength, flag);  
                ComputeY(j, this->tileLength, this->tileLength);  
                CopyOut(j, this->tileLength);
            }
            auto lengthY = this->blockLengthY - this->tileLength * (loopCountY - 1);
            auto cacLengthY = this->oriLengthY - this->tileLength * (loopCountY - 1);
            CopyInY(loopCountY - 1, lengthY, flag);  
            ComputeY(loopCountY - 1, this->tileLength, cacLengthY);  
            CopyOut(loopCountY - 1, lengthY);
            flag = false;
        }
        auto lengthX = this->blockLengthX - this->tileLength * (loopCountX - 1);
        auto cacLengthX = this->oriLengthX - this->tileLength * (loopCountX - 1);
        CopyInX(loopCountX - 1, lengthX);
        ComputeX(loopCountX - 1, cacLengthX, (lengthX + (this->ALIGN_NUM * 8) - 1) / (this->ALIGN_NUM * 8) * (this->ALIGN_NUM * 8));
        for (int32_t j = 0; j < loopCountY - 1; j ++) {
            CopyInY(j, this->tileLength, flag);  
            ComputeY(j, cacLengthX, this->tileLength);  
            CopyOut(j, this->tileLength);
        }
        auto lengthY = this->blockLengthY - this->tileLength * (loopCountY - 1);
        auto cacLengthY = this->oriLengthY - this->tileLength * (loopCountY - 1);
        CopyInY(loopCountY - 1, lengthY, flag);  
        ComputeY(loopCountY - 1, cacLengthX, cacLengthY);  
        CopyOut(loopCountY - 1, lengthY);
    }
private: 
    __aicore__ inline void CopyInX(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> xLocal = Q_x1.AllocTensor<TYPE_X1>();
        DataCopy(xLocal, Gm_x1[progress * this->tileLength], length);
        Q_x1.EnQue(xLocal);
    } 
    __aicore__ inline void CopyInY(int32_t progress, uint32_t length, bool isFirst = false) {
        LocalTensor<TYPE_Y> yLocal = Q_y.AllocTensor<TYPE_Y>();
        DataCopy(yLocal, Gm_y[progress * this->tileLength], length);
        if(isFirst) Duplicate(yLocal, TYPE_Y(0), length);
        Q_y.EnQue(yLocal);
    } 

    /*
        1. flag 标记 越界索引 i  （只用标记大的 小于min的自动处置为负了）
        2. 正常参与计算获取indices
        3. 根据 flag将 越界indices置为 -1 表示不存在
        4. 做indices 边界优化 
    */
    __aicore__ inline void ComputeX_1(int32_t progress, uint32_t length, uint32_t cmpLength) {
        LocalTensor<TYPE_X1> x = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<float> x1;
        if constexpr(std::is_same_v<T, half>) {
            auto fl_x = B_flx.Get<float>();
            Cast(fl_x, x, RoundMode::CAST_NONE, length);
            x1 = fl_x;
        } else if constexpr(std::is_same_v<T, int32_t>) {
            auto fl_x = B_flx.Get<float>();
            Cast(fl_x, x, RoundMode::CAST_NONE, length);
            x1 = fl_x;
        } else { // float
            x1 = x;
        }
        auto tmp = B_tmp.Get<float>();
        auto tmp2 = B_tmp2.Get<float>();
        auto maxBuffer = B_maxBuffer.Get<float>();
        auto minBuffer = B_minBuffer.Get<float>();
        auto bits = B_bits.Get<uint8_t>();
        Duplicate(maxBuffer, this->max, length);
        Duplicate(minBuffer, this->min, length);

        ReduceMax(tmp, x1, tmp2, length, false);
        float t_max = tmp.GetValue(0);
        ReduceMin(tmp, x1, tmp2, length, false);
        float t_min = tmp.GetValue(0);

        if (this->cacMinMax) {
            Duplicate(maxBuffer, t_max, length);
            Duplicate(minBuffer, t_min, length);
            this->cacMinMax = false;
        } else {
            Duplicate(tmp, t_max, length);
            Compare(bits, tmp, maxBuffer, CMPMODE::GT, cmpLength);
            if (bits.GetValue(0)) {
                Duplicate(maxBuffer, t_max, length);
            }
            
            Duplicate(tmp, t_min, length);
            Compare(bits, tmp, minBuffer, CMPMODE::LT, cmpLength);
            if (bits.GetValue(0)) {
                Duplicate(minBuffer, t_min, length);
            }
        }
        this->min = minBuffer.GetValue(0);
        this->max = maxBuffer.GetValue(0);
        Sub(maxBuffer, maxBuffer, minBuffer, 1);
        this->area = maxBuffer.GetValue(0);
        Q_x1.FreeTensor(x);
    }
    __aicore__ inline void ComputeX(int32_t progress, uint32_t length, uint32_t cmpLength) {
        LocalTensor<TYPE_X1> x = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<float> x1;
        if constexpr(std::is_same_v<T, half>) {
            auto fl_x = B_flx.Get<float>();
            Cast(fl_x, x, RoundMode::CAST_NONE, length);
            x1 = fl_x;
        } else if constexpr(std::is_same_v<T, int32_t>) {
            auto fl_x = B_flx.Get<float>();
            Cast(fl_x, x, RoundMode::CAST_NONE, length);
            x1 = fl_x;
        } else { // float
            x1 = x;
        }   
        auto indices = B_tmp.Get<float>();
        auto tmp = B_tmp2.Get<float>();
        auto bits = B_bits.Get<uint8_t>();
        auto transInt32Indices = B_transInt32Indices.Get<int32_t>();
        Duplicate(indices, min, length);
        Sub(indices, x1, indices, length); // indices = x - min
        
        Duplicate(tmp, this->area, length);
        Div(indices, indices, tmp, length); // indices = (x - min) / area   //归一分布
        Muls(indices, indices, float(bins), length); // indices = 对应bin下标
        Cast(indices, indices, RoundMode::CAST_FLOOR, length); // 向下处理一下


        // 处理越界
        Duplicate(tmp, float(max), length);
        Compare(bits, x1, tmp, CMPMODE::LE, cmpLength); // 小于等于max置为1
        Select(indices, bits, indices, float(-1.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        //[ 0. 3. -1. 3. -1.]
        
        Duplicate(tmp, float(bins), length);
        Compare(bits, tmp, indices, CMPMODE::GT, cmpLength); 
        Select(tmp, bits, this->zeros, float(-1.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Add(indices, indices, tmp, length); // 处于正好edge的边界 减一， 但是这里indices是处理过的 
        
        Cast(transInt32Indices, indices, RoundMode::CAST_FLOOR, length); // 将其取为int32 并用于 上下两个XY计算的沟通

        // Cast(int32Indices, indices, RoundMode::CAST_FLOOR, length);
        // for (uint32_t i = 0; i < length; i ++) {
        //     if (int32_t(int32Indices[i]) >= 0)
        //         tmp.SetValue(i, this->bin_edges[int32_t(int32Indices[i])]);
        //     else {} 
        //         // 为了不影响下compare 处于负数的应该都 对应this-zeros 或 减一都没问题所以不用改动 （因为仍小于等于0）
        // }   
        // Compare(bits, x1, tmp, CMPMODE::GE, cmpLength); // 大于等于当前对应边界的不用变，小于的减一
        // Select(tmp, bits, this->zeros, float(-1.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        // Add(indices, indices, tmp, length);

        // Cast(int32Indices, indices, RoundMode::CAST_FLOOR, length);
        // for (uint32_t i = 0; i < length; i ++) {
        //     int32_t j = int32Indices[i];
        //     if (j >= 0 && j != this->bins - 1)
        //         tmp.SetValue(i, this->bin_edges[j + 1]);
        //     else {
        //         tmp.SetValue(i, float(3e38))
        //     }// 为了不影响下compare 处于负数的应该都 对应this-zeros 防止 大于等于0 即 cmp 为 1， 即 x1的最大值
        // }
        // Compare(bits, x1, tmp, CMPMODE::LT, cmpLength);
        // Select(tmp, bits, this->zeros, float(1.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        // Add(indices, indices, tmp, length);
        Q_x1.FreeTensor(x);
    }
    __aicore__ inline void ComputeY(int32_t progress, uint32_t lengthX, uint32_t lengthY) {
        LocalTensor<TYPE_Y> y1 = Q_y.DeQue<TYPE_Y>();
        LocalTensor<float> y;
        if constexpr(std::is_same_v<T, half>) {
            auto fl_y = B_fly.Get<float>();
            Cast(fl_y, y1, RoundMode::CAST_NONE, lengthY);
            y = fl_y;

        } else if constexpr(std::is_same_v<T, int32_t>) {
            auto fl_y = B_fly.Get<float>();
            Cast(fl_y, y1, RoundMode::CAST_NONE, lengthY);
            y = fl_y;
            
        } else {
            y = y1;
        }

        auto transInt32Indices = B_transInt32Indices.Get<int32_t>();
        auto int32y = B_int32y.Get<int32_t>();  

        Cast(int32y, y, RoundMode::CAST_FLOOR, lengthY); // 每次从y中取出上次保存的值
        for (uint32_t i = 0; i < lengthX; i ++) { //transIn32Indices是由x的shape决定的 这个length 应为lengthX
            int32_t j = transInt32Indices.GetValue(i);
            if (j >= 0 && j < lengthY) {
                int32_t num = int32y.GetValue(j) + int32_t(1);
                int32y.SetValue(j, num);
            }
        }    
        Cast(y, int32y, RoundMode::CAST_NONE, lengthY); // 在重新转换为 y 
        Adds(transInt32Indices, transInt32Indices, -int32_t(lengthY), lengthY);

        if constexpr(std::is_same_v<T, half>) {
            Cast(y1, y, RoundMode::CAST_FLOOR, lengthY);
        } else if constexpr(std::is_same_v<T, int32_t>) {
            Cast(y1, y, RoundMode::CAST_ROUND, lengthY);
        } else { 
        }
        Q_y.EnQue(y1);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1;  
    TBuf<QuePosition::VECCALC> B_tmp, B_tmp2, B_bits, B_transInt32Indices, B_zeros, B_int32y, B_minBuffer, B_maxBuffer, B_flx, B_fly;  
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;  
    GlobalTensor<TYPE_X1> Gm_x1;       
    GlobalTensor<TYPE_Y> Gm_y;
    LocalTensor<float> zeros;

    uint32_t tileLength;
    uint32_t ALIGN_NUM;
    uint32_t tileNumX;
    uint32_t tileNumY;
    uint32_t totalLengthX;
    uint32_t totalLengthY;
    uint32_t blockLengthX; 
    uint32_t blockLengthY;
    uint32_t extraLengthX;
    uint32_t extraLengthY;
    uint32_t oriLengthX; 
    uint32_t oriLengthY;
    
    int32_t bins;
    float min;
    float max;
    float area;
    float* bin_edges;
    bool cacMinMax;
};


extern "C" __global__ __aicore__ void histogram(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    //GM_ADDR x1, GM_ADDR y, int32_t bins, float min, float max, float area, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain
    Histogram<DTYPE_X, DTYPE_Y> op;
    op.Init(x,y,tiling_data.cacMinMax, tiling_data.bins, tiling_data.min,tiling_data.max,
            tiling_data.area,tiling_data.ALIGN_NUM,tiling_data.block_size,
            tiling_data.totalLengthX,tiling_data.totalLengthY);
    if (tiling_data.cacMinMax) {
        op.PreProcess();
    }
     op.Process();
    
}